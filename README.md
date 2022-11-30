# GeneSegNet: A Deep Learning Framework for Cell Segmentation by Integrating Gene Expression and Imaging Information

## Overview
<div align=center><img src="https://github.com/BoomStarcuc/GeneSegNet/blob/master/data/GeneSegNet_framework.png" width="1000" height="390"/></div>

## Installation
1. Clone the repository, use:

``` 
git clone https://github.com/BoomStarcuc/GeneSegNet.git
```

2. Install dependencies, use:

```
pip install -r requirement.txt
```

## Datasets and Model

1. Download the demo datasets at [GoogleDrive](https://drive.google.com/drive/folders/1rF6U5fSq8D-UpZW-iUy4DG16dyxAzvK7?usp=share_link) and unzip them to your project directory.
2. Download GeneSegNet pre-trained model at [GoogleDrive](https://drive.google.com/drive/folders/1hzavxQ_zkH6At0vkCzskyg7hlRnKDEC3?usp=sharing), and put it into your project directory.

If you want to run the algorithm on your data , please see  **Training** part.

## Training from scratch
To run the algorithm on your data, use:

```
python -u GeneSeg_train.py --use_gpu --train_dir  training dataset path --val_dir validation dataset path --test_dir test dataset path --pretrained_model None --save_png --save_each --img_filter _image --mask_filter _label --all_channels --verbose --metrics --dir_above --save_model_dir save model path
```

Here:

- ```use_gpu``` will use gpu if torch with cuda installed.
- ```train_dir``` is a folder containing training data to train on.
- ```val_dir``` is a folder containing validation data to train on.
- ```test_dir``` is a folder containing test data to validate training results.
- ```img_filter```, ```mask_filter```, and ```heatmap_filter``` are end string for images, cell instance mask, and heat map.
- ```pretrained_model``` is a model to use for running or starting training.
- ```verbose``` shows information about running and settings and save to log.
- ```save_each``` save the model under per n epoch for later comparsion.
- ```save_png``` save masks as png and outlines as text file for ImageJ.
- ```metrics```compute the segmentation metrics.
- ```save_model_dir``` save training model to a directory

To see full list of command-line options run:

```
python GeneSeg_train.py --help
```
## Test

After trianing, to run test, use:

```
python GeneSegNet_test.py --use_gpu --test_dir test dataset path --pretrained_model your trained model --save_png --img_filter _image --mask_filter _label --all_channels --metrics --dir_above --output_filename a folder name
```

## Running a pre-trained model
Before running pre-trained model, please download pre-trained model provided. 

```
python GeneSegNet_test.py --use_gpu --test_dir test dataset path --pretrained_model pre-trained model --save_png --img_filter _image --mask_filter _label --all_channels --metrics --dir_above --output_filename a folder name
```

## Network Inference


