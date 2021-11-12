# SSD
An implementation of SSD object detection neural network using PyTorch.

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

2. Download pre-trained and fc-reduced `VGG16` model from [here](https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth).

3. Start training:

    ```shell
    conda activate SSD
    python train.py
    ```


## Reference
* [[Paper] SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)
* [[GitHub] amdegroot / ssd.pytorch](https://github.com/amdegroot/ssd.pytorch)

