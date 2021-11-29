# SSD
An implementation of SSD object detection using pytorch.

## Prepare environment
1. Create virtual environment:

    ```shell
    conda create -n SSD python=3.8
    conda activate SSD
    pip install -r requirements.txt
    ```

2. Install [pytorch](https://pytorch.org/), refer to the [blog](https://blog.csdn.net/qq_23013309/article/details/103965619) for details.


## Train
1. Download VOC2007 dataset from following website and unzip them:
   * http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
   * http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

2. Download pre-trained and fc-reduced `VGG16` model from [here](https://github.com/zhiyiYo/SSD/releases/download/v1.0.0/vgg16_reducedfc.pth).

3. Modify the value of `root` in `train.py`, please ensure that the directory structure of the `root` folder is as follows:

    ```txt
    root
    ├───Annotations
    ├───ImageSets
    │   ├───Layout
    │   ├───Main
    │   └───Segmentation
    ├───JPEGImages
    ├───SegmentationClass
    └───SegmentationObject
    ```

4. Start training:

    ```shell
    conda activate SSD
    python train.py
    ```

## Loss Curve
When `lr` 'is 5e-4, `batch_ size` is 8 and train on VOC2007 + VOC2012, the training loss curve is shown in following figure:

![损失曲线](resource/image/损失曲线.png)

## Evaluation
### one model
1. Modify the value of `root` and `model_path` in `eval.py`.
2. Calculate mAP:

    ```sh
    conda activate SSD
    python eval.py
    ```

### multi models
1. Modify the value of `root` and `model_dir` in `evals.py`.
2. Calculate and plot mAP:

    ```shell
    conda activate SSD
    python evals.py
    ```

### mAP of `SSD_120000.pth`

| class       | AP     |
| ----------- | ------ |
| aeroplane   | 73.65% |
| bicycle     | 83.03% |
| bird        | 70.77% |
| bottle      | 42.40% |
| bus         | 84.83% |
| car         | 84.64% |
| cat         | 89.66% |
| chair       | 56.04% |
| cow         | 80.18% |
| diningtable | 72.50% |
| dog         | 86.20% |
| horse       | 88.79% |
| motorbike   | 83.71% |
| person      | 78.56% |
| pottedplant | 42.70% |
| sheep       | 76.00% |
| sofa        | 77.81% |
| train       | 86.46% |
| tvmonitor   | 75.77% |
| mAP         | 74.93% |


## Detection
1. Download `SSD_120000.pth` from [here](https://github.com/zhiyiYo/SSD/releases/download/v1.1.0/SSD_120000.pth).
2. Modify the value of `model_path` and `image_path` in `demo.py`.
3. Display detection results:

    ```shell
    conda activate SSD
    python demo.py
    ```


## Notes
* Sometimes `loss` may become `nan`. If this happens, please reduce the value of `lr`.


## Reference
* [[Paper] SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)
* [[GitHub] amdegroot / ssd.pytorch](https://github.com/amdegroot/ssd.pytorch)
