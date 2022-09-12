# Gseg: A Deep Learning Framework for Cell Segmentation by Integrating Gene Expression and Imaging Information

## Overview


## Installation
1. Clone the repository, use:

``` 
git clone https://github.com/BoomStarcuc/Gseg.git
```

2. Install dependencies, use:

```
pip install -r requirement.txt
```

## Gseg Demo

1. Download the demo datasets at [GoogleDrive](https://drive.google.com/drive/folders/1OtppM5iinLMbZ5tlf8O6OuJMLqJq_p3M?usp=sharing) and unzip it to  ```datasets/```. These are a subset of the original images from the PciSeq datasets.
2. Download Gseg pre-trained model at [GoogleDrive](https://drive.google.com/drive/folders/1hzavxQ_zkH6At0vkCzskyg7hlRnKDEC3?usp=sharing), and put it into ```datasets/```
3. Open jupyter file ```Gseg_demo.ipynb``` in Gseg directory
4. Provide four section in this jupyter: 1. Load datasets； 2. Data preprocessing； 3. Training process； 4. Test using Gseg pre-trained model 

On the demo, we just provide a pipline of our Gseg. If you want to run the algorithm on your data , please see  **Training** part.

## Training from scratch
To run the algorithm on your data, use:

```
 python Gseg.py --use_gpu --dir original_image_path --img_filter _image --mask_filter _label --gaussian_filter _gaumap --pretrained_model None --verbose --save_each --save_png --variance 7 --save_dir save_path
```

Here:

- ```use_gpu``` will use gpu if torch with cuda installed.
- ```dir``` is a folder containing training data to train on.
- ```test_dir``` is a folder containing test data to validate training results.
- ```img_filter```, ```mask_filter```, and ```gaussian_filter``` are end string for images, cell instance mask, and heat map.
- ```pretrained_model``` is a model to use for running or starting training.
- ```verbose``` shows information about running and settings and save to log.
- ```save_each``` save the model under per n epoch for later comparsion.
- ```save_dir``` save preprocessing data to a directory

To see full list of command-line options run:

```
python train.py --help
```

## Running a pre-trained model

