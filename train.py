import sys, os, argparse, glob, pathlib, time
import subprocess
import cv2
import numpy as np
from natsort import natsorted
from tqdm import tqdm
import utils, metrics, models, io_img, core, plot
from pathlib import Path
import fastremap
import csv
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
import torch
from torch import nn    
import logging

parser = argparse.ArgumentParser(description='Gseg parameters')
    
# settings for CPU vs GPU
hardware_args = parser.add_argument_group("hardware arguments")
hardware_args.add_argument('--use_gpu', action='store_true', help='use gpu if torch with cuda installed')
hardware_args.add_argument('--gpu_device', required=False, default=0, type=int, help='which gpu device to use')
    
# settings for locating and formatting images
input_img_args = parser.add_argument_group("input image arguments")
input_img_args.add_argument('--dir',
                    default=[], type=str, help='folder containing data to run or train on.')
input_img_args.add_argument('--img_filter',
                    default='_image', type=str, help='end string for images to run on')
input_img_args.add_argument('--mask_filter',
                    default='_label', type=str, help='end string for masks to run on.')
input_img_args.add_argument('--gaussian_filter',
                    default='_gaumap', type=str, help='end string for gaussian map to run on.')
input_img_args.add_argument('--channel_axis',
                    default=None, type=int, help='axis of image which corresponds to image channels')
input_img_args.add_argument('--z_axis',
                    default=None, type=int, help='axis of image which corresponds to Z dimension')
input_img_args.add_argument('--chan',
                    default=0, type=int, help='channel to segment; 0: GRAY, 1: RED, 2: GREEN, 3: BLUE. Default: %(default)s')
input_img_args.add_argument('--chan2',
                    default=0, type=int, help='nuclear channel (if cyto, optional); 0: NONE, 1: RED, 2: GREEN, 3: BLUE. Default: %(default)s')
input_img_args.add_argument('--invert', action='store_true', help='invert grayscale channel')
# input_img_args.add_argument('--all_channels', action='store_true', help='use all channels in image if using own model and images with special channels')

# model settings 
model_args = parser.add_argument_group("model arguments")
model_args.add_argument('--pretrained_model', required=False, default='cyto', type=str, help='model to use for running or starting training')
# model_args.add_argument('--unet', action='store_true', help='run standard unet instead of cellpose flow output')
model_args.add_argument('--nclasses',default=3, type=int, help='if running unet, choose 2 or 3; cellpose always uses 3')

# algorithm settings
algorithm_args = parser.add_argument_group("algorithm arguments")
algorithm_args.add_argument('--no_resample', action='store_true', help="disable dynamics on full image (makes algorithm faster for images with large diameters)")
algorithm_args.add_argument('--net_avg', action='store_true', help='run 4 networks instead of 1 and average results')
algorithm_args.add_argument('--no_interp', action='store_true', help='do not interpolate when running dynamics (was default)')
algorithm_args.add_argument('--no_norm', action='store_true', help='do not normalize images (normalize=False)')
algorithm_args.add_argument('--do_3D', action='store_true', help='process images as 3D stacks of images (nplanes x nchan x Ly x Lx')
algorithm_args.add_argument('--diameter', required=False, default=32., type=float, 
                    help='cell diameter, if 0 will use the diameter of the training labels used in the model, or with built-in model will estimate diameter for each image')
algorithm_args.add_argument('--stitch_threshold', required=False, default=0.0, type=float, help='compute masks in 2D then stitch together masks with IoU>0.9 across planes')
algorithm_args.add_argument('--fast_mode', action='store_true', help='now equivalent to --no_resample; make code run faster by turning off resampling')

algorithm_args.add_argument('--flow_threshold', default=0.4, type=float, help='flow error threshold, 0 turns off this optional QC step. Default: %(default)s')
algorithm_args.add_argument('--cellprob_threshold', default=0, type=float, help='cellprob threshold, default is 0, decrease to find more and larger masks')

algorithm_args.add_argument('--anisotropy', required=False, default=1.0, type=float,
                    help='anisotropy of volume in 3D')
algorithm_args.add_argument('--exclude_on_edges', action='store_true', help='discard masks which touch edges of image')

# output settings
output_args = parser.add_argument_group("output arguments")
output_args.add_argument('--save_png', action='store_true', help='save masks as png and outlines as text file for ImageJ')
output_args.add_argument('--save_tif', action='store_true', help='save masks as tif and outlines as text file for ImageJ')
output_args.add_argument('--no_npy', action='store_true', help='suppress saving of npy')
output_args.add_argument('--savedir',
                    default=None, type=str, help='folder to which segmentation results will be saved (defaults to input image directory)')
output_args.add_argument('--dir_above', action='store_true', help='save output folders adjacent to image folder instead of inside it (off by default)')
output_args.add_argument('--in_folders', action='store_true', help='flag to save output in folders (off by default)')
output_args.add_argument('--save_flows', action='store_true', help='whether or not to save RGB images of flows when masks are saved (disabled by default)')
output_args.add_argument('--save_outlines', action='store_true', help='whether or not to save RGB outline images when masks are saved (disabled by default)')
output_args.add_argument('--save_ncolor', action='store_true', help='whether or not to save minimal "n-color" masks (disabled by default')
output_args.add_argument('--save_txt', action='store_true', help='flag to enable txt outlines for ImageJ (disabled by default)')
output_args.add_argument('--metrics', action='store_true', help='compute the segmentation metrics')
#output_args.add_argument('--variance', type=int, default=3)

# training settings
training_args = parser.add_argument_group("training arguments")
training_args.add_argument('--test_dir',
                    default=[], type=str, help='folder containing test data (optional)')
training_args.add_argument('--diam_mean',
                    default=32., type=float, help='mean diameter to resize cells to during training -- if starting from pretrained models it cannot be changed from 30.0')
training_args.add_argument('--learning_rate',
                    default=0.1, type=float, help='learning rate. Default: %(default)s')
training_args.add_argument('--weight_decay',
                    default=0.00001, type=float, help='weight decay. Default: %(default)s')
training_args.add_argument('--n_epochs',
                    default=500, type=int, help='number of epochs. Default: %(default)s')
training_args.add_argument('--batch_size',
                    default=4, type=int, help='batch size. Default: %(default)s')
training_args.add_argument('--min_train_masks',
                    default=1, type=int, help='minimum number of masks a training image must have to be used. Default: %(default)s')
training_args.add_argument('--residual_on',
                    default=1, type=int, help='use residual connections')
training_args.add_argument('--style_on',
                    default=1, type=int, help='use style vector')
training_args.add_argument('--concatenation',
                    default=0, type=int, help='concatenate downsampled layers with upsampled layers (off by default which means they are added)')
training_args.add_argument('--save_every',
                    default=100, type=int, help='number of epochs to skip between saves. Default: %(default)s')
training_args.add_argument('--save_each', action='store_true', help='save the model under a different filename per --save_every epoch for later comparsion')
training_args.add_argument('--variance', type=int, default=3)
training_args.add_argument('--gauAll', action='store_true')

# misc settings
parser.add_argument('--verbose', action='store_true', help='show information about running and settings and save to log')

def train():
    args = parser.parse_args()
    print("args:", args)

    if args.verbose:
        from io_img import logger_setup
        logger, log_file = logger_setup()
    else:
        print('No --verbose => no progress or info printed')
        logger = logging.getLogger(__name__)
                
    device, gpu = models.assign_device(use_torch=True, gpu=args.use_gpu, device=args.gpu_device)

    if args.pretrained_model is None or args.pretrained_model == 'None':
        pretrained_model = None
    else:
        pretrained_model = args.pretrained_model
        
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    output = io_img.load_train_test_data(args.dir, args.test_dir, args.img_filter, args.mask_filter, args.gaussian_filter)
    images, labels, gaumaps, gaumaps_all, label_names, test_images, test_labels, test_gaumaps, test_gaumaps_all, test_label_names = output

    # print("images_b:",np.array(images).shape) #(2239, 256, 256, 3)
    # print("labels_b:",np.array(labels).shape) #(2239, 256, 256)
    # print("gaumaps_b:",np.array(gaumaps).shape) #(2239, 256, 256)
    # print("gaumaps_all_b:",np.array(gaumaps_all).shape) #(2239, 256, 256)
    # print("test_images_b:",np.array(test_images).shape)#(560, 256, 256, 3)
    # print("test_labels_b:",np.array(test_labels).shape)#(560, 256, 256)
    # print("test_gaumaps_b:",np.array(test_gaumaps).shape)#(560, 256, 256)
    # print("test_gaumaps_all_b:",np.array(test_gaumaps_all).shape) #(560, 256, 256)

    #load images and gaussian maps for train
    images_gaumaps = []
    for n, (image, gaumap, gaumap_all) in enumerate(zip(images, gaumaps, gaumaps_all)):
        images_gaumaps.append(np.concatenate((image, gaumap[:,:,np.newaxis], gaumap_all[:,:,np.newaxis]), axis=2)) 
    
    images = np.array(images_gaumaps)

    #load images and gaussian maps for test
    images_gaumaps_test = []
    for n, (test_image, test_gaumap, test_gaumap_all) in enumerate(zip(test_images, test_gaumaps, test_gaumaps_all)):
        images_gaumaps_test.append(np.concatenate((test_image, test_gaumap[:,:,np.newaxis], test_gaumap_all[:,:,np.newaxis]), axis=2)) 
    
    test_images = np.array(images_gaumaps_test)

    logger.info('>>>> during training rescaling images to fixed diameter of %0.1f pixels'%args.diam_mean)
        
    model = models.CellposeModel(device=device,
                                pretrained_model=pretrained_model, 
                                diam_mean= args.diam_mean,
                                residual_on=args.residual_on,
                                style_on=args.style_on,
                                concatenation=args.concatenation,
                                nchan=4)

    cpmodel_path = model.train(images, labels, train_files=label_names,
                                test_data=test_images, test_labels=test_labels, test_files=test_label_names,
                                learning_rate=args.learning_rate, 
                                weight_decay=args.weight_decay,
                                channels=None,
                                save_path=os.path.realpath(args.dir), save_every=args.save_every,
                                save_each=args.save_each,
                                n_epochs=args.n_epochs,
                                batch_size=args.batch_size, 
                                min_train_masks=args.min_train_masks)
    model.pretrained_model = cpmodel_path
    logger.info('>>>> model trained and saved to %s'%cpmodel_path)

if __name__ == '__main__':
    train()
    
