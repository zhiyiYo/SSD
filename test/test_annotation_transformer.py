# coding:utf-8
import unittest
from net.dataset import AnnotationTransformer


class TestAnnotationTransformer(unittest.TestCase):
    """ 测试标签转换器 """

    def test_transform(self):
        """ 测试转换 """
        classes = [
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        class_to_index = {c: i for i, c in enumerate(classes)}
        
        transformer = AnnotationTransformer(class_to_index)
        file_path = 'data/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/Annotations/000002.xml'
        target = transformer(file_path)

        print('\n', target)
