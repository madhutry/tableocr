#@title IMport { form-width: "10%" }

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import os
import json
import os
import numpy as np
import json
from detectron2.structures import BoxMode
from   dataLocation import *
from detectron2.engine import DefaultTrainer

roots=getRoots()

def train():
  initData()
  cfg = get_cfg()
  cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
  cfg.DATASETS.TRAIN = ("table_trainval",)
  cfg.DATASETS.TEST = ()
  cfg.DATALOADER.NUM_WORKERS = 2
  cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
  cfg.SOLVER.IMS_PER_BATCH = 2
  cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
  cfg.SOLVER.CHECKPOINT_PERIOD = 500
  cfg.SOLVER.MAX_ITER = 1500    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
  cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
  cfg.OUTPUT_DIR="/content/detection/savedmodel/"
  trainer = DefaultTrainer(cfg) 
  trainer.resume_or_load(resume=True)
  trainer.train()


def initData():
  train_keys=["icdar","bank"]
  DatasetCatalog.clear()
  for d in ["trainval"]:
      DatasetCatalog.register("table_" + d, lambda d=d: get_icdar_dicts(train_keys))
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

