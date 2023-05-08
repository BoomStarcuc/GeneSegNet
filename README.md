# GeneSegNet: A Deep Learning Framework for Cell Segmentation by Integrating Gene Expression and Imaging Information. [BioRxiv](https://www.biorxiv.org/content/10.1101/2022.12.13.520283v1)

## Overview
<div align=center><img src="https://github.com/BoomStarcuc/GeneSegNet/blob/master/data/GeneSegNet_framework.png" width="1000" height="360"/></div>

## Installation
1. Create conda environments, use:

``` 
conda create -n GeneSegNet python=3.8
conda activate GeneSegNet
```

2. Install pytorch (1.12.1 Version), use:

``` 
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

but the above command may not match your CUDA environment, please check the link: https://pytorch.org/get-started/previous-versions/#v1121 to find the proper command that satisfied your CUDA environment.

3. Clone the repository, use:

``` 
git clone https://github.com/BoomStarcuc/GeneSegNet.git
```

4. Install dependencies, use:

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
- ```chan``` is a parameter to change the number of channels as input.
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

## Run a pre-trained model
Before running pre-trained model, please download pre-trained model provided. 

```
python GeneSegNet_test.py --use_gpu --test_dir test dataset path --pretrained_model pre-trained model --save_png --img_filter _image --mask_filter _label --all_channels --metrics --dir_above --output_filename a folder name
```

## Network Inference
To obtain final full-resolution segmentation results, use slidingwindows_gradient.py in Inference directory:

```
python slidingwindows_gradient.py
```

Note: root_dir, save_dir, and model_file need to be modified to your corresponding path.


