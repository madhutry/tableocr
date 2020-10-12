import detectron2.data
import dataset_mapper
import detection_utils
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from collections import OrderedDict
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
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import cv2
from detectron2.data.datasets import register_coco_instances
from dataset_mapper import TblDatasetMapper
import os


import logging
from detectron2.data import *
from dataset_mapper import TblDatasetMapper
def get_evaluator(cfg, dataset_name, output_folder=None):
    evaluator_list = []
    # evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    return COCOEvaluator(dataset_name, cfg, True, output_folder)


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        print('--',dataset_name)
        try:
          data_loader = build_detection_test_loader(cfg, dataset_name,mapper=TblDatasetMapper(cfg,is_train=False))
        except Exception as ex:
          logging.exception("Something awful happened!")
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def evaluate():
  train_keys=["bank"]
  DatasetCatalog.clear()
  MetadataCatalog.clear()
  keypoint_names = [ "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10" ]
  keypoint_connection_rules=[[ 1, 2, [255,175,100] ], [ 2, 3, [255,175,100] ], [ 3, 4,[255,175,100] ],[ 4, 5,[255,175,100] ], [ 5, 6,[255,175,100]], [ 6, 7,[255,175,100] ], [ 7, 8,[255,175,100] ], [ 8, 9,[255,175,100] ], [ 9, 10,[255,175,100] ] ] 
  keypoint_flip_map=[('c1','c10'),('c2','c9'),('c3','c8'),('c4','c7'),('c5','c6')]
  for d in ["testval"]:
      #DatasetCatalog.register("table_testval", lambda d=d: get_icdar_dicts(train_keys,'testjson'))
      register_coco_instances("table_testval", {}, "./annotations.json", "/content/VOC2007/JPEGImages")
      MetadataCatalog.get("table_testval").set(
        thing_classes=["table","r"],
        keypoint_names=keypoint_names,
        keypoint_connection_rules=keypoint_connection_rules
        ,keypoint_flip_map=keypoint_flip_map
        )

  cfg = get_cfg()
  cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
  cfg.DATALOADER.NUM_WORKERS = 2
  cfg.DATASETS.TRAIN = ("table_testval", )
  cfg.DATASETS.TEST = ("table_testval", )
  cfg.SOLVER.IMS_PER_BATCH = 2
  cfg.MODEL.DEVICE = "cpu"
  cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
  cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
  cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (ballon)
  cfg.MODEL.WEIGHTS = '/content/keypoints/workdir/savedmodel/model_final.pth'
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3   # set the testing threshold for this model
  cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 10
  model = build_model(cfg)
  DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=True)
  do_test(cfg, model)
