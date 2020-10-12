from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import torch
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import numpy as np
from matplotlib import pyplot as plt
import cv2
def predict(fn):
  cfg = get_cfg()
  cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
  cfg.DATALOADER.NUM_WORKERS = 2
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
  model.eval()
  checkpointer = DetectionCheckpointer(model)
  checkpointer.load(cfg.MODEL.WEIGHTS)
  imageori = cv2.imread(fn)
  # mywidth=600
  # img = Image.open(f4)
  # wpercent = (mywidth/float(img.size[0]))
  # hsize = int((float(img.size[1])*float(wpercent)))
  # imageori = np.array(img.resize((mywidth,hsize), PIL.Image.ANTIALIAS))
  with torch.no_grad():
    original_image = imageori[:, :, ::-1]
    height, width = imageori.shape[:2]
    image = torch.as_tensor(imageori.astype("float32").transpose(2, 0, 1))
    print('---',imageori.shape,height,width)
    inputs = {"image": image, "height": height, "width": width}
    outputs = model([inputs])[0]
    print(outputs["instances"].pred_classes)
    # print(outputs["instances"].pred_boxes)
    # print(outputs["instances"].pred_keypoints)
    # print(outputs["instances"].pred_keypoints.shape)
  MetadataCatalog.get("my_dataset_train").set(thing_classes=["table","r"])
  table_metadata = MetadataCatalog.get("my_dataset_train")
  v = Visualizer(imageori[:, :, ::-1],
                  metadata=table_metadata, 
                  scale=0.8
  )
  out = v.draw_instance_predictions(outputs["instances"])
  plt.figure(figsize=(100,100))
  plt.imshow(out.get_image()[:, :, ::-1])  