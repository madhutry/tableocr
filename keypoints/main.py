from detectron2.engine import DefaultTrainer
from TableTrainer import *
from dataset_mapper import *
from detection_utils import *
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import TableTrainer
import dataset_mapper
import detection_utils
import importlib
importlib.reload(TableTrainer)
importlib.reload(detection_utils)

importlib.reload(dataset_mapper)

from TableTrainer import *
from dataset_mapper import *
from detection_utils import *

#@title DataSet Loader { form-width: "10%" }

DatasetCatalog.clear()
MetadataCatalog.clear()
keypoint_names = [ "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10" ]
keypoint_connection_rules=[[ 1, 2, [255,175,100] ], [ 2, 3, [255,175,100] ], [ 3, 4,[255,175,100] ],[ 4, 5,[255,175,100] ], [ 5, 6,[255,175,100]], [ 6, 7,[255,175,100] ], [ 7, 8,[255,175,100] ], [ 8, 9,[255,175,100] ], [ 9, 10,[255,175,100] ] ] 
keypoint_flip_map=[('c1','c10'),('c2','c9'),('c3','c8'),('c4','c7'),('c5','c6')]
register_coco_instances("my_dataset_train", {}, "./annotations.json", "/content/VOC2007/JPEGImages")
my_md= MetadataCatalog.get("my_dataset_train").set(
    thing_classes=["table","r"],
    keypoint_names=keypoint_names,
    keypoint_connection_rules=keypoint_connection_rules
    ,keypoint_flip_map=keypoint_flip_map
    )
print('^^^^^^^^^^')    
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
#cfg.MODEL.WEIGHTS = '/content/workdir/keypoint2/model_0001999.pth'
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 15000    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.SOLVER.CHECKPOINT_PERIOD = 500

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 5   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (ballon)
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 10
cfg.OUTPUT_DIR="./workdir/savedmodel/"
trainer = TableTrainer(cfg) 
trainer.resume_or_load(resume=True)
trainer.train()

