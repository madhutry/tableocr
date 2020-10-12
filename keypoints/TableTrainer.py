from detectron2.engine import DefaultTrainer
from dataset_mapper import TblDatasetMapper
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.modeling import build_model
class TableTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
      return build_detection_train_loader(cfg, mapper=TblDatasetMapper(cfg))

    @classmethod
    def build_model(cls, cfg):
      model = build_model(cfg)
      return model