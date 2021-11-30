# coding:utf-8
from net import VOCDataset, TrainPipeline
from utils.augmentation_utils import SSDAugmentation


if __name__ == '__main__':
    # load dataset
    root = 'data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007'
    dataset = VOCDataset(root, 'trainval', SSDAugmentation(), True)

    # train config
    config = {
        'dataset': dataset,
        'n_classes': len(dataset.classes)+1,
        'vgg_path': 'model/vgg16_reducedfc.pth',
        'lr': 5e-4,
        'batch_size': 8,
        'num_workers': 4,
        'start_iter': 0,
        'max_iters': 120000,
        'warm_up_iters': 500,
        'lr_steps': (80000, 100000, 120000),
        'save_frequency': 5000,
    }

    # train
    train_pipeline = TrainPipeline(**config)
    train_pipeline.train()
