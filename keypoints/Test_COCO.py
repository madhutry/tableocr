from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
def drawCoco(imgid,catNm=['table','r']):
  dataType='val2017'
  annFile = "./annotations.json"
  coco=COCO(annFile)
  catIds = coco.getCatIds(catNms=catNm);
  imgIds = coco.getImgIds(catIds=catIds );
  imgIds = coco.getImgIds(imgIds = [imgid])
  img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
  f = '/content/VOC2007/JPEGImages/'+img['file_name']
  print(f)
  plt.figure(figsize=(25,25))
  I = io.imread(f)
  plt.imshow(I); plt.axis('off')
  ax = plt.gca()
  annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
  anns = coco.loadAnns(annIds)
  print('--',anns)
  coco.showAnns(anns)
  return img