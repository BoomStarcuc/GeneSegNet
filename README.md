# GeneSegNet: a deep learning framework for cell segmentation by integrating gene expression and imaging. [Genome Biology](https://link.springer.com/article/10.1186/s13059-023-03054-0?utm_source=rct_congratemailt&utm_medium=email&utm_campaign=oa_20231019&utm_content=10.1186/s13059-023-03054-0#availability-of-data-and-materials)

## Overview
<div align=center><img src="https://github.com/BoomStarcuc/GeneSegNet/blob/master/data/GeneSegNet_framework.png" width="1000" height="360"/></div>

## Installation
1. Create conda environments, use:

``` 
conda create -n GeneSegNet python=3.8
conda activate GeneSegNet
```

2. Install Pytorch (1.12.1 Version), use:

``` 
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

but the above command may not match your CUDA environment, please check the link: https://pytorch.org/get-started/previous-versions/#v1121 to find the proper command that satisfies your CUDA environment.

3. Clone the repository, use:

``` 
git clone https://github.com/BoomStarcuc/GeneSegNet.git
```

4. Install dependencies, use:

```
pip install -r requirement.txt
```

## Datasets and Model

1. Download the demo training datasets at [GoogleDrive](https://drive.google.com/drive/folders/1rF6U5fSq8D-UpZW-iUy4DG16dyxAzvK7?usp=share_link) and unzip them to your project directory.
2. Download GeneSegNet pre-trained model at [GoogleDrive](https://drive.google.com/drive/folders/1hzavxQ_zkH6At0vkCzskyg7hlRnKDEC3?usp=sharing), and put it into your project directory.


## Data preprocess

<h3 id="input-1">Input</h3>

Directory structure of initial input data. See hippocampus demo datasets at [GoogleDrive](https://drive.google.com/drive/folders/1n0cFQAcqGL3_wfGQtNLO_st7JQZEyd6h?usp=sharing).
```
your raw dataset
 |-images
 |   |-image sample 1
 |   |-image sample 2
 |   |-...
 |-labels
 |   |-label sample 1
 |   |-label sample 2
 |   |-...
 |-spots
 |   |-spot sample 1
 |   |-spot sample 2
 |   |-...
```

<h3 id="output-1">Output</h3>

After preprocessing, you will output a dataset without splitting into training, validation and testing, as follows：
```
your preprocessed dataset
 |-sample 1
 |   |-HeatMaps
 |   |   |-HeatMap
 |   |   |-HeatMap_all
 |   |-images
 |   |-labels
 |   |-spots
 |-sample 2
 |   |-HeatMaps
 |   |   |-HeatMap
 |   |   |-HeatMap_all
 |   |-images
 |   |-labels
 |   |-spots
 |-...
```

Please see preprocessed hippocampus demo datasets at [GoogleDrive](https://drive.google.com/drive/folders/1naUyOmtXLtK9-eAvl99mrcxmwJhP3ckT?usp=sharing).

<h3 id="code-run-1">Code run</h3>

If you use the demo training dataset we provided, you can skip this section. But if you want to train on your own dataset, you first need to run the preprocessing code in ```preprocess``` directory to satisfy the dataset structure during training.

```
python Generate_Image_Label_locationMap.py
```
Note: ```base_dir``` and ```save_crop_dir``` need to be modified to your corresponding path.


## Training from scratch

<h3 id="input-2">Input</h3>

You will need to split the output of the preprocessing step into training, validation, and test sets in reasonable proportions. The structure of the dataset should be as follows:
```
your split dataset
 |-train
 |   |-sample 1
 |   |   |-HeatMaps
 |   |   |   |-HeatMap
 |   |   |   |-HeatMap_all
 |   |   |-images            
 |   |   |-labels 
 |   |   |-spots
 |   |-sample 2
 |   |-...
 |-val
 |   |-sample 3
 |   |   |-HeatMaps
 |   |   |   |-HeatMap
 |   |   |   |-HeatMap_all
 |   |   |-images            
 |   |   |-labels 
 |   |   |-spots
 |   |-sample 4
 |   |-...
 |-test
 |   |-sample 5
 |   |   |-HeatMaps
 |   |   |   |-HeatMap
 |   |   |   |-HeatMap_all
 |   |   |-images            
 |   |   |-labels 
 |   |   |-spots
 |   |-sample 6
 |   |-...
```

Please see the demo training dataset at [GoogleDrive](https://drive.google.com/drive/folders/1rF6U5fSq8D-UpZW-iUy4DG16dyxAzvK7?usp=share_link). Then you can start to train your model using [command](#code-run-2). 

<h3 id="output-2">Output</h3>

After training, the algorithm will save the trained model to your specified path.

<h3 id="code-run-2">Code run</h3>

To run the algorithm on your data, use:

```
python -u GeneSeg_train.py --use_gpu --train_dir  training dataset path --val_dir validation dataset path --test_dir test dataset path --pretrained_model None --save_png --save_each --img_filter _image --mask_filter _label --all_channels --verbose --metrics --dir_above --save_model_dir save model path
```

Here:

- ```use_gpu``` will use GPU if torch with cuda installed.
- ```train_dir``` is a folder containing training data to train on.
- ```val_dir``` is a folder containing validation data to train on.
- ```test_dir``` is a folder containing test data to validate training results.
- ```img_filter```, ```mask_filter```, and ```heatmap_filter``` are end strings for images, cell instance mask, and heat map.
- ```pretrained_model``` is a model to use for running or starting training.
- ```chan``` is a parameter to change the number of channels as input (default 2 or 4).
- ```verbose``` shows information about running and settings and saves to log.
- ```save_each``` save the model under per n epoch for later comparison.
- ```save_png``` save masks as png and outlines as a text file for ImageJ.
- ```metrics``` compute the segmentation metrics.
- ```save_model_dir``` save training model to a directory

To see the full list of command-line options run:

```
python GeneSeg_train.py --help
```

## Test and run a pre-trained model

<h3 id="input-3">Input</h3>

The input is your test dataset.
```
your test dataset
 |-test
 |   |-sample 5
 |   |   |-HeatMaps
 |   |   |   |-HeatMap
 |   |   |   |-HeatMap_all
 |   |   |-images            
 |   |   |-labels 
 |   |   |-spots
 |   |-sample 6
 |   |-...
```

<h3 id="output-3">Output</h3>

The output will include the following two images: 1) the predicted cell instance masks; 2) the cell boundary comparison plot between predicted results and training labels.

<h3 id="code-run-3">Code run</h3>

To run the test or a pre-trained model, use:

```
python GeneSeg_test.py --use_gpu --test_dir test dataset path --pretrained_model your trained model --save_png --img_filter _image --mask_filter _label --all_channels --metrics --dir_above --output_filename a folder name
```

Note: if you want to run a pre-trained model, you should download the pre-trained model provided first. 

## Network Inference

<h3 id="input-4">Input</h3>

The input of the network inference is your raw datasets. See hippocampus demo datasets at [GoogleDrive](https://drive.google.com/drive/folders/1n0cFQAcqGL3_wfGQtNLO_st7JQZEyd6h?usp=sharing).

```
your raw dataset
 |-images
 |   |-image sample 1
 |   |-image sample 2
 |   |-...
 |-labels
 |   |-label sample 1
 |   |-label sample 2
 |   |-...
 |-spots
 |   |-spot sample 1
 |   |-spot sample 2
 |   |-...
```

<h3 id="output-4">Output</h3>

The output of the network inference includes four files of each sample as follows:
```
|-HeatMap
|   |-sample 1
|   |-sample 2
|- predicted full-resolution .mat file for sample 1
|- predicted full-resolution .png file for sample 1
|- predicted full-resolution .jpg file for sample 1
|- predicted full-resolution .mat file for sample 2
|- predicted full-resolution .png file for sample 2
|- predicted full-resolution .jpg file for sample 2
|-...
```

<h3 id="code-run-4">Code run</h3>

To obtain final full-resolution segmentation results, use slidingwindows_gradient.py in ```Inference``` directory:

```
python slidingwindows_gradient.py
```
Note: ```root_dir```, ```save_dir```, and ```model_file``` need to be modified to your corresponding path.

## Find the mapping relationships between transcripts and cells

<h3 id="input-5">Input</h3>

There are two types of input as follows:

```
1. your raw spot dataset
 |-spots
 |   |-spot sample 1
 |   |-spot sample 2
 |   |-...

2. your output of the network inference
 |-HeatMap
 |   |-sample 1
 |   |-sample 2
 |- predicted full-resolution .mat file for sample 1
 |- predicted full-resolution .png file for sample 1
 |- predicted full-resolution .jpg file for sample 1
 |- predicted full-resolution .mat file for sample 2
 |- predicted full-resolution .png file for sample 2
 |- predicted full-resolution .jpg file for sample 2
 |-...
```

<h3 id="output-5">Output</h3>

The output is a .csv file including four columns (```cell_id```, ```spotX```, ```spotY```, and ```gene```) so that each gene will find its unique corresponding cell.

```
   cell_id   spotX   spotY   gene
 |    0	      213     419	   Pvalb
 |    0	      248     442	   Gad1
 |    1	      1212    18	    Plp1
 |    .        .       .      .
 |    .        .       .      .
 |    .        .       .      .
```

<h3 id="code-run-5">Code run</h3>

```
python generate_MappingRelationships.py
```
Note: ```spot_dir```, ```label_dir```, and ```save_dir``` need to be modified to your corresponding path.

## Citation
If you find our work useful for your research, please consider citing the following paper.
```
@article{wang2023genesegnet,
  title={GeneSegNet: a deep learning framework for cell segmentation by integrating gene expression and imaging},
  author={Wang, Yuxing and Wang, Wenguan and Liu, Dongfang and Hou, Wenpin and Zhou, Tianfei and Ji, Zhicheng},
  journal={Genome Biology},
  volume={24},
  number={1},
  pages={235},
  year={2023},
  publisher={Springer}
}
```
