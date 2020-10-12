import detectron2
from pycocotools.coco import COCO

from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import os, json, cv2, random
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
from detectron2.structures import BoxMode

def showGt(imgid,catNm):
  dataType='val2017'
  annFile = "./annotations.json"
  coco=COCO(annFile)
  catIds = coco.getCatIds(catNms=catNm);
  imgIds = coco.getImgIds(catIds=catIds );
  imgIds = coco.getImgIds(imgIds = [imgid])
  img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
  annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
  anns = coco.loadAnns(annIds)
  DatasetCatalog.clear()
  MetadataCatalog.clear()
  keypoint_names_col = [ "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10" ]
  keypoint_names_row= [ 'r'+str(i) for i in range(10)]
  keypoint_names=keypoint_names_col+keypoint_names_row
  print(keypoint_names)
  register_coco_instances("my_dataset_train", {}, "./annotations.json", "/content/VOC2007/JPEGImages")
  my_md= MetadataCatalog.get("my_dataset_train").set(
      thing_classes=["table","r"],
      #keypoint_names=keypoint_names,
      #keypoint_connection_rules=[]
      #,keypoint_flip_map=[]
      )
  f = '/content/VOC2007/JPEGImages/'+img['file_name']
  img = cv2.imread(f)
  visualizer = Visualizer(img[:, :, ::-1], metadata=my_md)
  for an in anns:
    an["bbox_mode"]=BoxMode.XYWH_ABS
  d = {'annotations':anns}
  out = visualizer.draw_dataset_dict(d)
  return out.get_image()[:, :, ::-1]
  #cv2_imshow()