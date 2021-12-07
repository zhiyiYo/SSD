# coding:utf-8
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2 as cv
from imgaug import augmenters as iaa
from imgaug.augmentables import BoundingBox, BoundingBoxesOnImage

from net import CustomDataset


rot90 = iaa.Rot90(keep_size=False)
dataset = CustomDataset('data/Hotspot', 'traintest')


for image_path, anno_path in zip(dataset.image_paths, dataset.annotation_paths):

    # 读入图像
    image = cv.imread(image_path)
    boxes = []

    dom = ET.parse(anno_path)
    root = dom.getroot()
    for obj in root.iter('object'):
        x1 = int(obj.find('bndbox/xmin').text)
        y1 = int(obj.find('bndbox/ymin').text)
        x2 = int(obj.find('bndbox/xmax').text)
        y2 = int(obj.find('bndbox/ymax').text)
        boxes.append(BoundingBox(x1, y1, x2, y2))

    boxes = BoundingBoxesOnImage(boxes, image.shape)

    # 上下翻转图像和边界框
    image_, bbox_ = rot90(image=image, bounding_boxes=boxes)
    bbox_ = bbox_.to_xyxy_array(int)

    # 新的图片和标签文件
    image_path = Path(image_path)
    anno_path = Path(anno_path)
    image_path_ = image_path.with_name(image_path.stem+'_rot90.jpg')
    anno_path_ = anno_path.with_name(anno_path.stem+'_rot90.xml')

    # 修改标签文件信息
    root.find('filename').text = image_path_.name
    root.find('path').text = str(image_path_.absolute())
    root.find('size/width').text = str(image_.shape[1])
    root.find('size/height').text = str(image_.shape[0])
    root.find('size/depth').text = str(image_.shape[2])

    for i, obj in enumerate(root.iter('object')):
        box = bbox_[i].astype(str)
        obj.find('bndbox/xmin').text = box[0]
        obj.find('bndbox/ymin').text = box[1]
        obj.find('bndbox/xmax').text = box[2]
        obj.find('bndbox/ymax').text = box[3]

    cv.imwrite(str(image_path_), image_)
    tree_ = ET.ElementTree(root)
    tree_.write(anno_path_, encoding='utf-8')

