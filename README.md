# Gseg: A deep learning framework for cell segmentation by integrating gene expression and imaging information

## Overview


## Installation
1. Clone the repository.
2. install dependencies.

```
pip install -r requirement.txt
```

## Gseg Demo

1. Download the google drive [GoogleDrive](https://drive.google.com/drive/folders/1OtppM5iinLMbZ5tlf8O6OuJMLqJq_p3M?usp=sharing) and unzip it to  ```datasets/```. These are a subset of the original images from the PciSeq datasets.
2. Download face Gseg pre-trained model at [GoogleDrive](https://drive.google.com/drive/folders/1hzavxQ_zkH6At0vkCzskyg7hlRnKDEC3?usp=sharing), and put it into ```datasets/```
3. Open jupyter file ```Gseg_demo.ipynb``` in Gseg directory
4. provide four section in this jupyter: 1. Load Datasets； 2. Data preprocessing； 3. training process； 4. Test using Gseg pre-trained model 

On the demo, we just provide a pipline of our Gseg. If you want to run the algorithm on your data , please see  **Training** part.

# Training from scratch
To run the algorithm on your data, use:

```
 python train.py --use_gpu --dir training data path --test_dir test data path --img_filter _image --mask_filter _label --pretrained_model None --verbose --save_each
```

Here:

- ```use_gpu``` will use gpu if torch with cuda installed.
- ```dir``` is a folder containing training data to train on.
- ```test_dir``` is a folder containing test data to validate training results.
- ```img_filter``` and mask_filter are end string for images and cell instance mask.
- ```pretrained_model``` is a model to use for running or starting training.
- ```verbose``` shows information about running and settings and save to log.
- ```save_each``` save the model under per 100 epoch for later comparsion.

To see full list of command-line options run:

```
python train.py --help
```

## Running a pre-trained model

