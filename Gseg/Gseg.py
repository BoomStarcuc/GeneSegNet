import sys, os, argparse, glob, pathlib, time
import subprocess
import cv2
import numpy as np
from natsort import natsorted
from tqdm import tqdm
import utils, metrics, models, io_img, core, plot, preprocess
from pathlib import Path
import fastremap
import csv
import h5py
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
                    default=["C:/Users/booms/Downloads/Gseg/datasets/original_data"], required = True, type=str, help='folder containing original data to be processed.')
input_img_args.add_argument('--train_dir',
                    default=[], type=str, help='folder containing data to run or train on.')
input_img_args.add_argument('--test_dir',
                    default=[], type=str, help='folder containing test data')
# input_img_args.add_argument('--spot_dir',
#                     default=[], type=str, help='folder containing gene spot data')
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
input_img_args.add_argument('--invert', action='store_true', help='invert grayscale channel')

# model settings 
model_args = parser.add_argument_group("model arguments")
model_args.add_argument('--pretrained_model', required=False, default=None, type=str, help='model to use for running or starting training')
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
output_args.add_argument('--output_filename', default="output", type=str, help='output filename')                    
output_args.add_argument('--dir_above', action='store_true', help='save output folders adjacent to image folder instead of inside it (off by default)')
output_args.add_argument('--in_folders', action='store_true', help='flag to save output in folders (off by default)')
output_args.add_argument('--save_flows', action='store_true', help='whether or not to save RGB images of flows when masks are saved (disabled by default)')
output_args.add_argument('--save_outlines', action='store_true', help='whether or not to save RGB outline images when masks are saved (disabled by default)')
output_args.add_argument('--save_ncolor', action='store_true', help='whether or not to save minimal "n-color" masks (disabled by default')
output_args.add_argument('--save_txt', action='store_true', help='flag to enable txt outlines for ImageJ (disabled by default)')
# output_args.add_argument('--metrics', action='store_true', help='compute the segmentation metrics')


preprocessing_args = parser.add_argument_group("preprocessing srguments")
preprocessing_args.add_argument('--variance', type=int, default=7)
preprocessing_args.add_argument('--roi_size', default=[256,256], help='The window size to execute SlidingWindow inference')
preprocessing_args.add_argument('--sw_batch-size', type=int, default=10)
preprocessing_args.add_argument('--save_dir', default=["C:/Users/booms/Downloads/Gseg/datasets/training_test_data"], required = True, type=str, help='save preprocessing data to a directory')

# training settings
training_args = parser.add_argument_group("training arguments")
training_args.add_argument('--diam_mean',
                    default=34., type=float, help='mean diameter to resize cells to during training -- if starting from pretrained models it cannot be changed from 30.0')
training_args.add_argument('--learning_rate',
                    default=0.1, type=float, help='learning rate. Default: %(default)s')
training_args.add_argument('--weight_decay',
                    default=0.00001, type=float, help='weight decay. Default: %(default)s')
training_args.add_argument('--n_epochs',
                    default=5, type=int, help='number of epochs. Default: %(default)s')
training_args.add_argument('--batch_size',
                    default=4, type=int, help='batch size. Default: %(default)s')
training_args.add_argument('--min_train_masks',
                    default=2, type=int, help='minimum number of masks a training image must have to be used. Default: %(default)s')
training_args.add_argument('--residual_on',
                    default=1, type=int, help='use residual connections')
training_args.add_argument('--style_on',
                    default=1, type=int, help='use style vector')
training_args.add_argument('--concatenation',
                    default=0, type=int, help='concatenate downsampled layers with upsampled layers (off by default which means they are added)')
training_args.add_argument('--save_every',
                    default=1, type=int, help='number of epochs to skip between saves. Default: %(default)s')
training_args.add_argument('--save_each', action='store_true', help='save the model under a different filename per --save_every epoch for later comparsion')


# misc settings
parser.add_argument('--verbose', action='store_true', help='show information about running and settings and save to log')

def train(args, logger, N):            
    device, gpu = models.assign_device(use_torch=True, gpu=args.use_gpu, device=args.gpu_device)

    if args.pretrained_model is None or args.pretrained_model == 'None':
        pretrained_model = None
    else:
        pretrained_model = args.pretrained_model

    output = io_img.load_train_test_data(args.train_dir, N, args.test_dir, args.img_filter, args.mask_filter, args.gaussian_filter, args.output_filename, args.variance)
    images, labels, gaumaps, gaumaps_all, label_names, test_images, test_labels, test_gaumaps, test_gaumaps_all, test_label_names = output

    # print("images_b:",np.array(images).shape) #(2239, 256, 256, 3)
    # print("labels_b:",np.array(labels).shape) #(2239, 256, 256)
    # print("gaumaps_b:",np.array(gaumaps).shape) #(2239, 256, 256)
    # print("gaumaps_all_b:",np.array(gaumaps_all).shape) #(2239, 256, 256)
    # print("test_images_b:",np.array(test_images).shape)#(560, 256, 256, 3)
    # print("test_labels_b:",np.array(test_labels).shape)#(560, 256, 256)
    # print("test_gaumaps_b:",np.array(test_gaumaps).shape)#(560, 256, 256)
    # print("test_gaumaps_all_b:",np.array(test_gaumaps_all).shape) #(560, 256, 256)

    assert len(images) == len(labels) == len(gaumaps) == len(gaumaps_all) == len(label_names)
    assert len(test_images) == len(test_labels) == len(test_gaumaps) == len(test_gaumaps_all) == len(test_label_names)

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

    logger.info('>>>> during training rescaling images to fixed diameter of %0.1f pixels'% args.diam_mean)
    print("device:", device)
    print("pretrained_model:", pretrained_model)
    print("args.diam_mean:", args.diam_mean)
    print("args.residual_on:", args.residual_on)
    print("args.style_on:", args.style_on)
    print("args.concatenation", args.concatenation)
    print("label_names:", label_names)
    print("test_label_names:", test_label_names)
    print("args.learning_rate", args.learning_rate)
    print("args.weight_decay:", args.weight_decay)
    print("args.save_every:", args.save_every)
    print("args.dir:", args.dir)
    print("args.n_epochs:", args.n_epochs)
    print("args.batch_size:", args.batch_size)
    print("args.min_train_masks:", args.min_train_masks) 

    model = models.GsegModel(gpu=args.use_gpu, device=device,
                                pretrained_model=pretrained_model, 
                                diam_mean= args.diam_mean,
                                residual_on=args.residual_on,
                                style_on=args.style_on,
                                concatenation=args.concatenation,
                                nchan=4)

    model_path = model.train(images, labels, train_files=label_names,
                                test_data=test_images, test_labels=test_labels, test_files=test_label_names,
                                learning_rate=args.learning_rate, 
                                weight_decay=args.weight_decay,
                                channels=None,
                                save_path=os.path.realpath(args.dir), save_every=args.save_every,
                                save_each=args.save_each,
                                n_epochs=args.n_epochs,
                                batch_size=args.batch_size, 
                                min_train_masks=args.min_train_masks)
                                
    model.pretrained_model = model_path
    logger.info('>>>> model trained and saved to %s'%model_path)
    print("model_path:", model_path)
    args.pretrained_model = model_path

def test(args, logger, N):
    tic = time.time()

    total_test_dir =  args.test_dir+args.train_dir 
    print("total_test_dir:", total_test_dir)

    #load image, label, gaumap
    output = io_img.load_train_test_data(total_test_dir, N-1, image_filter = args.img_filter, 
                                    mask_filter = args.mask_filter, gaussian_filter = args.gaussian_filter, 
                                    foldername = args.output_filename, variance = args.variance)
    
    images, labels, gaumaps, gaumaps_all, label_names,_,_,_,_,_ = output

    #load gene spots
    spot_names_list = []
    for each_test_dir in total_test_dir:
        spot_names = natsorted(glob.glob(os.path.join(os.path.join(each_test_dir, 'spots'), '*.csv')))
        spot_names_list.append(np.array(spot_names))
    spot_names = np.concatenate(spot_names_list, axis=0)

    print("images_t:",np.array(images).shape) #(2239, 256, 256, 3)
    print("labels_t:",np.array(labels).shape) #(2239, 256, 256)
    print("gaumaps_t:",np.array(gaumaps).shape) #(2239, 256, 256)
    print("gaumaps_all_t:",np.array(gaumaps_all).shape) #(2239, 256, 256)
    print("spot_names_t:", np.array(spot_names).shape)

    pretrained_model =  args.pretrained_model
    saving_something = args.save_png or args.save_tif or args.save_flows or args.save_ncolor or args.save_txt
    device, gpu = models.assign_device(use_torch=True, gpu=args.use_gpu, device=args.gpu_device)
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("@@@@@@@@@@@@@@@@@@pretrained_model@@@@@@@@@@@@@@@@:", pretrained_model)
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    model = models.GsegModel(gpu=gpu, device=device, 
                                    pretrained_model=pretrained_model,
                                    diam_mean=args.diam_mean,
                                    residual_on=args.residual_on,
                                    style_on=args.style_on,
                                    concatenation=args.concatenation,
                                    net_avg=False,
                                    nchan=4)

    logger.info('>>>> using diameter %0.3f for all images'% args.diameter)

    filename = []
    masks_true = []
    masks_pred = []
    #tqdm_out = utils.TqdmToLogger(logger,level=logging.INFO)
    
    assert len(images) == len(labels) == len(gaumaps) == len(gaumaps_all) == len(spot_names) == len(label_names)
    
    for image, label, gaumap, label_name, spot_name in tqdm(zip(images, labels, gaumaps, label_names, spot_names)):
        # print("&&&&&&&&&&&&&&&label_name&&&&&&&&&&&&&&&&&&&&:", label_name)
        # print("&&&&&&&&&&&&&&&label_name&&&&&&&&&&&&&&&&&&&&:", str(Path(label_name).parent.parent.absolute()))
        # print("&&&&&&&&&&&&&&& test_dir &&&&&&&&&&&&&&&&&&&&:", args.test_dir)
        # print("&&&&&&&&&&&&&&& IS &&&&&&&&&&&&&&&&&&&&:", str(Path(label_name).parent.parent.absolute()) in args.test_dir)

        with open(spot_name, 'r') as f:
            lines = f.readlines()
            lines = lines[1:]

        spot_list = []
        for line in lines:
            splits = line.strip().split(',')
            if splits[0] != '':
                spot_list.append([int(float(splits[0])), int(float(splits[1]))])
        
        spot = np.array(spot_list) # (N,2)

        image_gaumap = np.concatenate((image, gaumap[:,:,np.newaxis]), axis=2)

        out = model.eval(image_gaumap, diameter=args.diam_mean,
                        do_3D=args.do_3D, net_avg=(not args.fast_mode or args.net_avg),
                        augment=False,
                        resample=(not args.no_resample and not args.fast_mode),
                        flow_threshold = args.flow_threshold,
                        cellprob_threshold = args.cellprob_threshold,
                        stitch_threshold = args.stitch_threshold,
                        invert = args.invert,
                        batch_size = args.batch_size,
                        interp = (not args.no_interp),
                        normalize = (not args.no_norm),
                        channel_axis = args.channel_axis,
                        z_axis = args.z_axis,
                        anisotropy = args.anisotropy,
                        model_loaded = True)
        masks, flows = out[:2]
        label = label[0]

        if args.exclude_on_edges:
            masks = utils.remove_edge_masks(masks)
        if args.no_npy:
            io_img.masks_flows_to_seg(image, masks, flows, diams, label_name, channels)
        if saving_something:
            io_img.save_masks(image, masks, flows, label, gaumap, spot, label_name, N, args.variance, 
                            foldername = args.output_filename, png=args.save_png, tif=args.save_tif,
                            save_flows=args.save_flows,save_outlines=args.save_outlines,
                            save_ncolor=args.save_ncolor,dir_above=args.dir_above,savedir=args.savedir,
                            save_txt=args.save_txt, in_folders=args.in_folders)
        
        # print("label:", label.shape, label.dtype) #float32
        # print("masks:", masks.shape, masks.dtype)  #uint16
        if label.max() != 0 and str(Path(label_name).parent.parent.absolute()) in args.test_dir:
            filename.append(os.path.basename(label_name))
            masks_true.append(label.astype(np.uint16))
            masks_pred.append(masks.astype(np.uint16))

    ap, tp, fp, fn = metrics.average_precision(masks_true, masks_pred)
    print("average precision: %0.3f at threshold 0.5, %0.3f at threshold 0.75 and %0.3f at threshold 0.9"%(np.mean(ap[:,0]), np.mean(ap[:,1]), np.mean(ap[:,2])))

    logger.info('>>>> completed in %0.3f sec'%(time.time()-tic))
    return ap

def Gseg():
    args = parser.parse_args()
    print("args:", args)

    #preprocessing original data
    image_dirs = natsorted(glob.glob(os.path.join(os.path.join(args.dir, 'image', '*.jpg'))))
    label_dirs = natsorted(glob.glob(os.path.join(os.path.join(args.dir, 'label', '*.mat'))))
    spot_dirs = natsorted(glob.glob(os.path.join(os.path.join(args.dir, 'spots', '*.csv'))))

    print("image_dirs, label_dirs, spot_dirs:", len(image_dirs), len(label_dirs), len(spot_dirs))

    #parameter configuration for sliding window and gene heatmap
    # Device configuration
    device, gpu = models.assign_device(use_torch=True, gpu=args.use_gpu, device=args.gpu_device)

    assert len(image_dirs) == len(label_dirs) == len(spot_dirs)

    data_dir_list = []
    for image_dir, label_dir, spot_dir in zip(image_dirs, label_dirs, spot_dirs):
        #load original image
        image = cv2.imread(image_dir)
        image = image[..., [2,1,0]] #convert BGR into RGB

        #Load cell instance mask
        label = h5py.File(label_dir, 'r')
        label = np.array(label['CellMap']).transpose()

        #Load gene expressions
        with open(spot_dir, 'r') as f:
            lines = f.readlines()
            lines = lines[1:]

        spot_list = []
        for line in lines:
            splits = line.strip().split(',')
            spot_list.append([splits[0], splits[1], splits[2]])
                
        spot = np.array(spot_list)

        # print("image:", image.shape)
        # print("label:", label.shape)
        # print("spot:", spot.shape)

        each_slice_path = preprocess.preprocessing(image, label, spot, args.variance, 
                                                    args.roi_size, args.sw_batch_size, 
                                                    args.save_dir, image_dir, return_type=1, device=device)
        
        data_dir_list.append(str(Path(each_slice_path).absolute()))
    
    #print("data_dir_list:", data_dir_list)
    
    #split data into training data and test data
    split_index = int(len(data_dir_list) * 2 / 3)
    args.train_dir = data_dir_list[:split_index]
    args.test_dir = data_dir_list[split_index:]

    print("args.train_dir:", args.train_dir)
    print("args.test_dir:", args.test_dir)
    #training
    if args.verbose:
        from io_img import logger_setup
        logger, log_file = logger_setup()
    else:
        print('No --verbose => no progress or info printed')
        logger = logging.getLogger(__name__)
    
    IoU = 0
    N = 1
    while IoU <= 0.99:
        print("{}th external iteration".format(N))
        train(args, logger, N-1)
        IoUs = test(args, logger, N)
        IoU = np.mean(IoUs[0])
        N += 1
        print("IoU:", IoU)

if __name__ == '__main__':
    Gseg()
    
