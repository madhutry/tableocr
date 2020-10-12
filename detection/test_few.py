from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from collections import OrderedDict
from google.colab.patches import cv2_imshow

from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
)
from detectron2.evaluation import (
    COCOEvaluator,
    verify_results,
    inference_on_dataset,
    print_csv_format,
)

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from   dataLocation import *
import os, json, cv2, random
from detectron2.structures import BoxMode

roots=getRoots()


def initData():
  train_keys=["icdar","bank"]
  DatasetCatalog.clear()
  for d in ["testval"]:
      DatasetCatalog.register("table_" + d, lambda d=d: get_icdar_dicts(train_keys,'testjson'))
      MetadataCatalog.get("table_" + d).set(thing_classes=["border"])

def get_icdar_dicts(keys,jsonkey="trainjson"):
    allimages={}
    for key in keys:
      root_dir = roots[key]['root']
      img_dir= os.path.join(root_dir,roots[key]['image'])
      trains_jsons = roots[key][jsonkey]
      for trn_json in trains_jsons:
        json_file=os.path.join(root_dir,trn_json+'.json')
        with open(json_file) as f:
            imgs_anns = json.load(f)
            for img in imgs_anns['images']:
              filename = os.path.join(img_dir, img["file_name"])
              if not os.path.exists(filename):
                continue
              record = {}
              idx = img['id']
              height, width = cv2.imread(filename).shape[:2]
              record["file_name"] = filename
              record["image_id"] = idx
              record["height"] = height
              record["width"] = width
              record["annotations"]=[]
              allimages[trn_json+'_'+str(img['id'])]=record
            for anno in imgs_anns['annotations']:
              idx = trn_json+'_'+str(anno['image_id'])
              catId = anno['category_id']
              if catId==2:
                #catId=1
                continue
              elif catId==1:
                catId=0 #Bordered
              elif catId==3:
                catId=0 #Borderless
              if idx not in allimages.keys():
                continue
              record = allimages[idx]
              obj = {
                    "bbox": anno['bbox'],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation":[],
                    "category_id": catId,
                }
              record['annotations'].append(obj)
    f = [t for t in allimages.values() if len(t['annotations'])>0]
    return f


def test():
  initData()
  train_keys=["icdar","bank"]
  cfg = get_cfg()
  cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
  cfg.DATASETS.TRAIN = ("table_trainval",)
  cfg.DATASETS.TEST = ()
  cfg.DATALOADER.NUM_WORKERS = 2
  cfg.SOLVER.IMS_PER_BATCH = 2
  cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
  cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
  cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)

  cfg.MODEL.WEIGHTS = '/content/detection/savedmodel/model_final.pth'
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6   # set the testing threshold for this model
  cfg.DATASETS.TEST = ("table_trainval", )
  predictor = DefaultPredictor(cfg)
  dataset_dicts = get_icdar_dicts(train_keys,'testjson')
  MetadataCatalog.get("table_bank").set(thing_classes=["table"])
  table_metadata = MetadataCatalog.get("table_bank")
  for d in random.sample(dataset_dicts,4):    
      print(d["file_name"])
      im = cv2.imread(d["file_name"])
      #im = cv2.imread("/content/datacocotable/Cell_images/cTDaR_t10535.jpg")
      outputs = predictor(im)
      v = Visualizer(im[:, :, ::-1],
                    metadata=table_metadata, 
                    scale=0.8
      )
      out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
      cv2_imshow(out.get_image()[:, :, ::-1])    
