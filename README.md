# SSD
An implementation of SSD object detection using PyTorch.

## Quick start
1. Create virtual environment:

    ```shell
    conda create -n SSD python=3.8
    conda activate SSD
    pip install -r requirements.txt
    ```

2. Install [PyTorch](https://pytorch.org/), refer to the [blog](https://blog.csdn.net/qq_23013309/article/details/103965619) for details.


## Train
1. Download VOC2007 dataset:

    ```shell
    cd data
    python download.py
    ```

2. Download pre-trained and fc-reduced `VGG16` model from [here](https://github.com/zhiyiYo/SSD/releases/download/v1.0.0/vgg16_reducedfc.pth).

3. Start training:

    ```shell
    conda activate SSD
    python train.py
    ```

## Evaluation
1. Modify the `model_path` in `eval.py`.
2. Calculate mAP:

    ```shell
    conda activate SSD
    python eval.py
    ```


## Detection
1. Modify the `model_path` and `image_path` in `demo.py`.

2. Display detection results:

    ```shell
    conda activate SSD
    python demo.py
    ```


## Notes
1. Sometimes the data set downloaded through `download.py`' may be incomplete, so please check whether the number of pictures in the data set is correct after downloading, or you can download the data set directly through the browser to the following two addresses:
   * http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
   * http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar


## Reference
* [[Paper] SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)
* [[GitHub] amdegroot / ssd.pytorch](https://github.com/amdegroot/ssd.pytorch)
