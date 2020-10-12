#!/usr/bin/env python

import argparse
import collections
import datetime
import glob
import json
import os
import os.path as osp
import sys
import uuid

import imgviz
import numpy as np

import labelme

try:
    import pycocotools.mask
except ImportError:
    print("Please install pycocotools:\n\n    pip install pycocotools\n")
    sys.exit(1)

class args:
    input_dir='/content/VOC2007/JPEGImages/'
    #input_dir='/mldata/banks/'

    output_dir='/mldata/banks/results'
    labels='/content/keypoints/labeltxt.txt'
    noviz=False

def pointInRect(point,rect):
    x1, y1, w, h = rect
    x2, y2 = x1+w, y1+h
    x, y = point
    if (x1 < x and x < x2):
        if (y1 < y and y < y2):
            return True
    return False
def main():
    os.makedirs(args.output_dir,exist_ok=True)
    os.makedirs(osp.join(args.output_dir, "JPEGImages"),exist_ok=True)
    if not args.noviz:
        os.makedirs(osp.join(args.output_dir, "Visualization"),exist_ok=True)
    print("Creating dataset:", args.output_dir)

    now = datetime.datetime.now()

    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ),
        licenses=[dict(url=None, id=0, name=None,)],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type="instances",
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )

    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        print(line,i)
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        if class_id == -1:
            continue
        class_name_to_id[class_name] = class_id
        keypointsNm = ['c'+str(no) for no in range(1,11,1)]
        skeleton=[[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10]]
        data["categories"].append(
            dict(supercategory=None, id=class_id, keypoints=keypointsNm ,name=class_name,skeleton=skeleton,)
        )
    out_ann_file = osp.join(args.output_dir, "annotations.json")
    print('op',out_ann_file)
    label_files = glob.glob(osp.join(args.input_dir, "*.json"))
    for image_id, filename in enumerate(label_files):
        print("Generating dataset from:", filename)

        label_file = labelme.LabelFile(filename=filename)

        base = osp.splitext(osp.basename(filename))[0]
        out_img_file = osp.join(args.output_dir, "JPEGImages", base + ".jpg")

        img = labelme.utils.img_data_to_arr(label_file.imageData)
        imgviz.io.imsave(out_img_file, img)
        data["images"].append(
            dict(
                license=0,
                url=None,
                file_name=osp.basename(osp.relpath(out_img_file, osp.dirname(out_ann_file))),
                height=img.shape[0],
                width=img.shape[1],
                date_captured=None,
                id=image_id,
            )
        )

        masks = {}  # for area
        segmentations = collections.defaultdict(list)  # for segmentation
        allckeypoints=[]
        allrkeypoints=[]
        for shape in label_file.shapes:
            points = shape["points"]
            label = shape["label"]
            group_id = shape.get("group_id")
            shape_type = shape.get("shape_type", "polygon")
            mask = labelme.utils.shape_to_mask(
                img.shape[:2], points, shape_type
            )

            if group_id is None:
                group_id = uuid.uuid1()

            instance = (label, group_id)

            if instance in masks:
                masks[instance] = masks[instance] | mask
            else:
                masks[instance] = mask

            if shape_type == "rectangle":
                (x1, y1), (x2, y2) = points
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                points = [x1, y1, x2, y1, x2, y2, x1, y2]
            elif shape_type == "point":
                (x1, y1) = points[0]
                if label == 'c':
                    allckeypoints.append([x1,y1])
                if label == 'r':
                    allrkeypoints.append([x1,y1])
            else:
                points = np.asarray(points).flatten().tolist()
            segmentations[instance].append(points)
        segmentations = dict(segmentations)
        for instance, mask in masks.items():
            cls_name, group_id = instance
            if cls_name not in class_name_to_id:
                continue
            cls_id = class_name_to_id[cls_name]

            mask = np.asfortranarray(mask.astype(np.uint8))
            mask = pycocotools.mask.encode(mask)
            area = float(pycocotools.mask.area(mask))
            bbox = pycocotools.mask.toBbox(mask).flatten().tolist()
            ckeypoints=[]
            for kp in allckeypoints:
                if pointInRect(kp,bbox):
                    ckeypoints.append([kp[0],kp[1],2])
            # rkeypoints=[]
            # for kp in allrkeypoints:
            #     if pointInRect(kp,bbox):
            #         rkeypoints.append([kp[0],kp[1],2])

            ckeypoints=np.array(sorted(ckeypoints, key=lambda x: x[0], reverse=False))
            # rkeypoints=np.array(sorted(rkeypoints, key=lambda y: y[1], reverse=False))

            c_num_keypoints=int(ckeypoints.shape[0])
            # r_num_keypoints=int(rkeypoints.shape[0])
            if c_num_keypoints>0:
                keypoints_empty= np.zeros((10,3))
                keypoints_empty[0:ckeypoints.shape[0],0:ckeypoints.shape[1]]=ckeypoints
                ckeypoints=keypoints_empty.ravel().tolist()
            else:
                keypoints_empty= np.zeros((10,3))
                ckeypoints=keypoints_empty.ravel().tolist()
            # if r_num_keypoints>0:
            #     keypoints_empty=b = np.zeros((25,3))
            #     keypoints_empty[0:rkeypoints.shape[0],0:rkeypoints.shape[1]]=rkeypoints
            #     rkeypoints=keypoints_empty.ravel().tolist()
            # else:
            #     rkeypoints=[]      
            # keypoints = np.hstack((ckeypoints,rkeypoints)).tolist()
            keypoints = ckeypoints
            kdict = dict(
                id=len(data["annotations"]),
                image_id=image_id,
                category_id=cls_id,
                segmentation=segmentations[instance],
                area=area,
                bbox=bbox,
                iscrowd=0,
                #r_num_keypoints=r_num_keypoints
            )
            if len(keypoints)>0:
                kdict['keypoints']=keypoints
                kdict['c_num_keypoints']=c_num_keypoints
            data["annotations"].append(kdict)

        if not args.noviz:
            labels, captions, masks = zip(
                *[
                    (class_name_to_id[cnm], cnm, msk)
                    for (cnm, gid), msk in masks.items()
                    if cnm in class_name_to_id
                ]
            )
            viz = imgviz.instances2rgb(
                image=img,
                labels=labels,
                masks=masks,
                captions=captions,
                font_size=15,
                line_width=2,
            )
            out_viz_file = osp.join(
                args.output_dir, "Visualization", base + ".jpg"
            )
            imgviz.io.imsave(out_viz_file, viz)
    with open(out_ann_file, "w") as f:
        json.dump(data, f)
    print(out_ann_file)
    


if __name__ == "__main__":
    main()
