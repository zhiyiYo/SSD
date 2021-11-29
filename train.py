# coding:utf-8
from net import VOCDataset, TrainPipeline
from utils.augmentation_utils import SSDAugmentation


if __name__ == '__main__':
    # load dataset
    root = 'data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007'
    dataset = VOCDataset(root, 'trainval', SSDAugmentation(), True)

    # train
    train_pipeline = TrainPipeline(
        dataset,
        vgg_path='model/vgg16_reducedfc.pth',
        batch_size=8,
        num_workers=4,
        n_classes=21,
        lr=2.5e-4,
        lr_steps=(80000, 100000, 120000),
        max_iter=120000,
        save_frequency=5000,
        warm_up_iters=500
    )
    train_pipeline.train()
