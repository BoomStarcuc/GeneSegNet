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
input_img_args.add_argument('--look_one_level_down', action='store_true', help='run processing on all subdirectories of current folder')
input_img_args.add_argument('--img_filter',
                    default=[], type=str, help='end string for images to run on')
input_img_args.add_argument('--channel_axis',
                    default=None, type=int, help='axis of image which corresponds to image channels')
input_img_args.add_argument('--z_axis',
                    default=None, type=int, help='axis of image which corresponds to Z dimension')
input_img_args.add_argument('--chan',
                    default=0, type=int, help='channel to segment; 0: GRAY, 1: RED, 2: GREEN, 3: BLUE. Default: %(default)s')
input_img_args.add_argument('--chan2',
                    default=0, type=int, help='nuclear channel (if cyto, optional); 0: NONE, 1: RED, 2: GREEN, 3: BLUE. Default: %(default)s')
input_img_args.add_argument('--invert', action='store_true', help='invert grayscale channel')
input_img_args.add_argument('--all_channels', action='store_true', help='use all channels in image if using own model and images with special channels')

# model settings 
model_args = parser.add_argument_group("model arguments")
model_args.add_argument('--pretrained_model', required=False, default='cyto', type=str, help='model to use for running or starting training')
model_args.add_argument('--unet', action='store_true', help='run standard unet instead of cellpose flow output')
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
training_args.add_argument('--train', action='store_true', help='train network using images in dir')
training_args.add_argument('--test_dir',
                    default=[], type=str, help='folder containing test data (optional)')
training_args.add_argument('--mask_filter',
                    default='_masks', type=str, help='end string for masks to run on. Default: %(default)s')
training_args.add_argument('--diam_mean',
                    default=32., type=float, help='mean diameter to resize cells to during training -- if starting from pretrained models it cannot be changed from 30.0')
training_args.add_argument('--learning_rate',
                    default=0.1, type=float, help='learning rate. Default: %(default)s')
training_args.add_argument('--weight_decay',
                    default=0.00001, type=float, help='weight decay. Default: %(default)s')
training_args.add_argument('--n_epochs',
                    default=500, type=int, help='number of epochs. Default: %(default)s')
training_args.add_argument('--batch_size',
                    default=8, type=int, help='batch size. Default: %(default)s')
training_args.add_argument('--min_train_masks',
                    default=2, type=int, help='minimum number of masks a training image must have to be used. Default: %(default)s')
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

def main():
    args = parser.parse_args()
    print("args:", args)

    if args.verbose:
        from io_img import logger_setup
        logger, log_file = logger_setup()
    else:
        print('>>>> !NEW LOGGING SETUP! To see cellpose progress, set --verbose')
        print('No --verbose => no progress or info printed')
        logger = logging.getLogger(__name__)

    # find images
    if len(args.img_filter)>0:
        imf = args.img_filter
    else:
        imf = None
    
    if not args.train:
        saving_something = args.save_png or args.save_tif or args.save_flows or args.save_ncolor or args.save_txt
                
    device, gpu = models.assign_device(use_torch=True, gpu=args.use_gpu, device=args.gpu_device)

    if args.pretrained_model is None or args.pretrained_model == 'None':
        pretrained_model = False
    else:
        pretrained_model = args.pretrained_model
        
    if not args.train:
        tic = time.time()

        image_names = io.get_image_files(args.dir, 
                                            args.mask_filter, 
                                            imf=imf,
                                            look_one_level_down=args.look_one_level_down)

        label_names, flow_names = io.get_label_files(image_names, args.mask_filter, imf=args.img_filter)

        root = str(Path(args.dir).parent.absolute())
        #gaumap_root = str(Path(args.dir).parent.absolute()) + '/gauMap_Var{}'.format(args.variance)
        gaumap_root = str(Path(args.dir).parent.absolute()) + '/gauMap_VarUNet{}'.format(args.variance)
        print("root:",root)
        

        if args.gauAll:
            Gaumap_dir = os.path.join(gaumap_root, 'GauMap_all')
        else:
            Gaumap_dir = os.path.join(gaumap_root, 'GauMap')
        
        print("gaumap_root:",Gaumap_dir)

        #mask_dir = os.path.join(root, 'mask/')
        spot_dir = os.path.join(root, 'spot/')
        #watershed_dir = os.path.join(root, 'watershed/')

        #mask_names = natsorted(glob.glob(os.path.join(mask_dir, '*.jpg')))
        spot_names = natsorted(glob.glob(os.path.join(spot_dir, '*.csv')))
        gaumap_names = natsorted(glob.glob(os.path.join(Gaumap_dir, '*.jpg')))
        #watershed_names = natsorted(glob.glob(os.path.join(watershed_dir, '*.jpg')))

        #print("main_mask_names:",mask_names)
        #print("main_mask_names:",len(mask_names))
        #print("main_spot_names:",spot_names)
        #print("main_spot_names:",len(spot_names))
        #print("main_image_names:",image_names)
        # print("main_image_names:",len(image_names))
        #print("main_label_names:",label_names)
        # print("main_label_names:",len(label_names))
        #print("main_gaumap_names:",gaumap_names)

        nimg = len(image_names)
            
        cstr0 = ['GRAY', 'RED', 'GREEN', 'BLUE']
        cstr1 = ['NONE', 'RED', 'GREEN', 'BLUE']
        logger.info('>>>> running cellpose on %d images using chan_to_seg %s and chan (opt) %s'%
                        (nimg, cstr0[channels[0]], cstr1[channels[1]]))
        
        # handle built-in model exceptions; bacterial ones get no size model 
        if builtin_size:
            model = models.Cellpose(gpu=gpu, device=device, model_type=model_type, 
                                            net_avg=(not args.fast_mode or args.net_avg))
            
        else:
            # if args.all_channels:
            #     channels = None
                
            if args.all_channels:
                nchan = 4
                channels = None 
            else:
                nchan = 2

            szmean = args.diam_mean
            # print("args.residual_on:",args.residual_on) #1
            # print("args.concatenation:",args.concatenation)#0
            # print("nchan:",nchan) # 3
            # print("args.style_on:",args.style_on) #1 
                    
            pretrained_model = None if model_type is not None else pretrained_model

            # print("gpu:",gpu) # False
            # print("device:", device) #cpu
            # print("pretrained_model:", pretrained_model) # a path
            # print("model_type:", model_type) #None
            # print("diam_mean:", szmean) #34.0
            # print("args.residual_on:",args.residual_on) # 1
            # print("args.style_on:",args.style_on) # 1
            # print("args.concatenation:",args.concatenation)# 0
            # print("nchan:",nchan) # 4
                

            model = models.CellposeModel(gpu=gpu, device=device, 
                                            pretrained_model=pretrained_model,
                                            model_type=model_type,
                                            diam_mean=szmean,
                                            residual_on=args.residual_on,
                                            style_on=args.style_on,
                                            concatenation=args.concatenation,
                                            net_avg=False,
                                            nchan=nchan)
        
        # handle diameters
        if args.diameter==0:
            if builtin_size:
                diameter = None
                logger.info('>>>> estimating diameter for each image')
            else:
                logger.info('>>>> not using cyto, cyto2, or nuclei model, cannot auto-estimate diameter')
                diameter = model.diam_labels
                logger.info('>>>> using diameter %0.3f for all images'%diameter)
        else:
            diameter = args.diameter
            logger.info('>>>> using diameter %0.3f for all images'%diameter)
        
        filename = []
        masks_true = []
        masks_pred = []
        cell_calling_metric = []
        density_metrics = []
        tqdm_out = utils.TqdmToLogger(logger,level=logging.INFO)

        # print("image_names:", len(image_names))
        # print("label_names:", len(label_names))
        # print("flow_names:", len(flow_names))
        # print("gaumap_names:", len(gaumap_names))
        # print("mask_names:", len(mask_names))
        # print("spot_names:", len(spot_names))
        #assert len(image_names) == len(label_names) == len(flow_names) == len(gaumap_names) == len(mask_names) == len(spot_names)
        assert len(image_names) == len(label_names) == len(flow_names) == len(gaumap_names) == len(spot_names)
        #for image_name, label_name, flow_name, gaumap_name, mask_name, spot_name in tqdm(zip(image_names, label_names, flow_names, gaumap_names, mask_names, spot_names), file=tqdm_out):
        for image_name, label_name, flow_name, gaumap_name, spot_name in tqdm(zip(image_names, label_names, flow_names, gaumap_names, spot_names), file=tqdm_out):
            print("main_img_file:", image_name)
            print("main_label_file:", label_name)
            print("main_flow_name:", flow_name)
            print("main_gaumap_name;", gaumap_name)
            #print("main_mask_name:", mask_name)
            print("main_spot_name:", spot_name)
            print("----------------------------------------------------------------------------")
            image = io.imread(image_name)
            label = io.imread(label_name)
            flow = io.imread(flow_name)

            gaumap = cv2.imread(gaumap_name,0)

            # print("@@@@@@@@@@@@image@@@@@@@@@@@@@:", image.shape)#(256,256,3)
            # print("@@@@@@@@@@@@gaumap@@@@@@@@@@@@@:", gaumap.max()) #(256,256)
            # print("@@@@@@@@@@@@gaumap@@@@@@@@@@@@@:", gaumap)

            image_gaumap = np.concatenate((image, gaumap[:,:,np.newaxis]), axis=2)

            # mask_area = cv2.imread(mask_name,0)
            # mask_area = mask_area/255
            # mask_area[mask_area >= 0.5] = 1
            # mask_area[mask_area < 0.5] = 0

            with open(spot_name, 'r') as f:
                lines = f.readlines()
                lines = lines[1:]

            spot_list = []
            for line in lines:
                splits = line.split(',')
                spot_list.append([int(float(splits[0])), int(float(splits[1]))])
            
            spot = np.array(spot_list)

            # print("main_image:",image.shape)  #[256,256,4]
            # print("main_img_max:", np.max(image))
            # print("main_img_min:", np.min(image))
            # print("main_image_gaumap:",image_gaumap.shape)  #[256,256,4]
            # print("main_image_gaumap_max:", np.max(image_gaumap))
            # print("main_image_gaumap_min:", np.min(image_gaumap))
            # print("main_gaumap:",gaumap.shape)  #[256,256]
            # print("main_gaumap_max:", np.max(gaumap))
            # print("main_gaumap_min:", np.min(gaumap))
            # print("main_label:",label.shape) #[256,256]
            # print("main_label_max:", np.max(label))
            # print("main_label_min:", np.min(label))

            # print("main_flow:",flow.shape) # [4,256,256]
            # print("main_flow_max:", np.max(flow))
            # print("main_flow_min:", np.min(flow))

            # print("image_gaumap:", image_gaumap.shape) #[256,256,4]
            # print("channels:", channels) #None
            # print("diameter:", diameter) #32
            # print("args.do_3D:", args.do_3D) # False
            # print("net_avg:", (not args.fast_mode or args.net_avg)) #True
            # print("resample:", (not args.no_resample and not args.fast_mode)) #0.4
            # print("flow_threshold:", args.flow_threshold) #0
            # print("cellprob_threshold:", args.cellprob_threshold) #0.0
            # print("stitch_threshold:", args.stitch_threshold) # False
            # print("invert:", args.invert) #False
            # print("batch_size:", args.batch_size) # 8
            # print("interp:", (not args.no_interp)) # True
            # print("normalize:", (not args.no_norm)) # True
            # print("channel_axis:", args.channel_axis) #None
            # print("z_axis:", args.z_axis) #None
            # print("anisotropy", args.anisotropy) #1.0


            out = model.eval(image_gaumap, channels=channels, diameter=diameter,
                            do_3D=args.do_3D, net_avg=(not args.fast_mode or args.net_avg),
                            augment=False,
                            resample=(not args.no_resample and not args.fast_mode),
                            flow_threshold=args.flow_threshold,
                            cellprob_threshold=args.cellprob_threshold,
                            stitch_threshold=args.stitch_threshold,
                            invert=args.invert,
                            batch_size=args.batch_size,
                            interp=(not args.no_interp),
                            normalize=(not args.no_norm),
                            channel_axis=args.channel_axis,
                            z_axis=args.z_axis,
                            anisotropy=args.anisotropy,
                            model_loaded=True)
            masks, flows = out[:2]


            # print("masks_pred:", masks.shape) #[256,256]
            # print("masks_pred_max:", np.max(masks))
            # print("masks_pred_min:", np.min(masks))
            # print("masks_flows:", np.array(flows).shape)
            # print("flows[1]_dp:", flows[1].shape)

            # dp = flows[1]
            # cellprop = flows[2]
            # print("dp:",np.transpose(dp, (1,2,0)).shape)
            # print("cellprop:", cellprop[..., None].shape)
            # computemask = np.concatenate((np.transpose(dp, (1,2,0)), cellprop[..., None]), axis=2)
            # mdict = {'CellMap': computemask}
            # savemat("computemask.mat", mdict)

            # print("v_flow:", flows[1][0][None, ...].repeat(2,axis=0).shape)
            # print("cellprop:", flows[2].shape)
            #v = cv2.applyColorMap((flows[1][0]).astype(np.uint8), cv2.COLORMAP_JET)
            # v_flow = plot.dx_to_circ(flows[1][0][None, ...].repeat(2,axis=0))
            # v_flow[v_flow==0] = 255
            # cv2.imwrite('/home/yw2009/anaconda3/envs/medicalimage/cellposes/pipline_data/v_gradient.jpg', v_flow)

            #h = cv2.applyColorMap((flows[1][1]).astype(np.uint8), cv2.COLORMAP_JET)
            # h_flow = plot.dx_to_circ(flows[1][1][None, ...].repeat(2,axis=0))
            # h_flow[h_flow==0] = 255
            # cv2.imwrite('/home/yw2009/anaconda3/envs/medicalimage/cellposes/pipline_data/h_gradient.jpg', h_flow)

            # cellprop = (flows[2]>0.5).astype(np.uint8)*255
            # cv2.imwrite('/home/yw2009/anaconda3/envs/medicalimage/cellposes/pipline_data/levelset3_binary.jpg', cellprop)

            # plt.imshow(flows[2])
            # plt.axis('off')
            # plt.savefig('/home/yw2009/anaconda3/envs/medicalimage/cellposes/pipline_data/Cellprop.jpg', bbox_inches='tight', pad_inches = 0 )
            # plt.clf()

            # binarylabel = label.copy()
            # binarylabel[binarylabel!=0] =255
            # cv2.imwrite('/home/yw2009/anaconda3/envs/medicalimage/cellposes/pipline_data/binary.jpg', binarylabel)

            if len(out) > 3:
                diams = out[-1]
            else:
                diams = diameter
            if args.exclude_on_edges:
                masks = utils.remove_edge_masks(masks)
            if args.no_npy:
                io.masks_flows_to_seg(image, masks, flows, diams, image_name, channels)
            if saving_something:
                io.save_masks(image, masks, flows, label, gaumap, spot, image_name, args.gauAll, args.variance, png=args.save_png, tif=args.save_tif,
                                save_flows=args.save_flows,save_outlines=args.save_outlines,
                                save_ncolor=args.save_ncolor,dir_above=args.dir_above,savedir=args.savedir,
                                save_txt=args.save_txt,in_folders=args.in_folders)

            
            # prop, isCount = cell_calling(masks, mask_area, spot)
            # if isCount:
            #     cell_calling_metric.append(prop)

            # density, isCount = density_metric(masks, spot)
            # if isCount:
            #     density_metrics.extend(density)
            
            if label.max() != 0:
                #print("basename:", os.path.basename(image_name))
                filename.append(os.path.basename(image_name))
                masks_true.append(label)
                masks_pred.append(masks)

        print("main_len_filename:", np.array(filename).shape)
        print("main_len_masks_true:", np.array(masks_true).shape)
        print("main_len_masks_pred:", np.array(masks_pred).shape)
        # print("main_len_density_metrics:", len(density_metrics))
        if metrics:
            ap, tp, fp, fn = metrics.average_precision(masks_true, masks_pred)
            precision, recall, fscore = metrics.boundary_scores(masks_true, masks_pred, [1])
            aji = metrics.aggregated_jaccard_index(masks_true, masks_pred)
            # avg_cell_calling = np.mean(cell_calling_metric)
            # avg_density_metric = np.mean(np.array(density_metrics))

            #print("average precision:", ap.shape) #[334,3]
            #print("precision:", precision.shape) #[1,334]
            #print("aji:", aji.shape) #[334,]

            # boundary_metrics = np.transpose(np.concatenate((precision, recall, fscore), axis=0)) #[334, 3]
            # csvdata = np.concatenate((np.array(filename)[:,np.newaxis], ap, boundary_metrics, aji[:,np.newaxis]), axis=1)
            # print("csvdata:",csvdata.shape)

            # #print("saved csv path:", root + "/gauMap_Var7_UNet_lr0.1_diam34.csv")

            # with open(root + "/gauMap_Var3_All_lr0.1.csv", "w") as csvfile:
            #     writer = csv.writer(csvfile)
            #     writer.writerow(['filename', '0.5 ap', '0.75 ap', '0.9 ap', 'precision', 'recall', 'fscore', 'AJI'])
            #     writer.writerows(csvdata)

            precision_mask = ~np.isnan(precision)
            recall_mask = ~np.isnan(recall)
            fscore_mask = ~np.isnan(fscore)

            #print("mask:", precision_mask.shape, recall_mask.shape, fscore_mask.shape)
            score_mask = precision_mask * recall_mask * fscore_mask

            print("average precision: %0.3f at threshold 0.5, %0.3f at threshold 0.75 and %0.3f at threshold 0.9"%(np.mean(ap[:,0]), np.mean(ap[:,1]), np.mean(ap[:,2])))
            print("boundary scores: precision : %0.3f, recall : %0.3f, fscore : %0.3f"%(np.mean(precision[score_mask]), np.mean(recall[score_mask]), np.mean(fscore[score_mask])))
            print("aggregated jaccard index: %0.3f"%(np.mean(aji)))
            # print("cell calling: %0.3f and density: %f"%(avg_cell_calling, avg_density_metric))

        logger.info('>>>> completed in %0.3f sec'%(time.time()-tic))
    else:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print("args.test_dir:",args.test_dir) #/hpc/group/jilab/imaging/Yuxing/cellpose/test_pciSeq
        # print("args.unet:",args.unet) #Flase
        test_dir = None if len(args.test_dir)==0 else args.test_dir

        output = io.load_train_test_data(args.dir, test_dir, imf, args.mask_filter, args.unet, args.look_one_level_down)
        images, labels, image_names, test_images, test_labels, image_names_test = output

        # print("images_b:",np.array(images).shape) #(2239, 256, 256, 3)
        # print("labels_b:",np.array(labels).shape) #(2239, 4, 256, 256)
        # print("test_images_b:",np.array(test_images).shape)#(560, 256, 256, 3)
        # print("test_labels_b:",np.array(test_labels).shape)#(560, 4, 256, 256)

        # root = str(Path(args.dir).parent.absolute()) + '/gauMap_Var{}'.format(args.variance)
        # test_root = str(Path(test_dir).parent.absolute()) + '/gauMap_Var{}'.format(args.variance)

        root = str(Path(args.dir).parent.absolute()) + '/gauMap_VarUNet{}'.format(args.variance)
        test_root = str(Path(test_dir).parent.absolute()) + '/gauMap_VarUNet{}'.format(args.variance)
        

        #if args.gauAll:
        Gaumap_dir_all = os.path.join(root, 'GauMap_all')
        Gaumap_test_dir_all = os.path.join(test_root, 'GauMap_all')
        #else:
        Gaumap_dir = os.path.join(root, 'GauMap')
        Gaumap_test_dir = os.path.join(test_root, 'GauMap')

        print("Gaumap_dir_all:", Gaumap_dir_all)
        print("Gaumap_test_dir_all:", Gaumap_test_dir_all)
        print("main_Gaumap_dir:", Gaumap_dir)
        print("main_Gaumap_test_dir:", Gaumap_test_dir)

        gaumap_names_all = natsorted(glob.glob(os.path.join(Gaumap_dir_all, '*.jpg')))
        gaumap_names_test_all = natsorted(glob.glob(os.path.join(Gaumap_test_dir_all, '*.jpg')))
        gaumap_names = natsorted(glob.glob(os.path.join(Gaumap_dir, '*.jpg')))
        gaumap_names_test = natsorted(glob.glob(os.path.join(Gaumap_test_dir, '*.jpg')))
        
        #print("image_names:",image_names[::200])
        #print("Gaumap_names:",gaumap_names[::200])
        #print("image_names:",len(image_names))
        #print("Gaumap_names:",len(gaumap_names))
        # print("image_names:",image_names_test[::200])
        # print("Gaumap_names:",gaumap_names_test[::200])
        # print("image_names:",len(image_names_test))
        # print("Gaumap_names:",len(gaumap_names_test))

        ######load images and gaussian maps for train
        gaumaps = []
        gaumaps_all = []
        images_gaumaps = []
        for n, (image_name, gaumap_name, gaumap_name_all) in enumerate(zip(image_names, gaumap_names, gaumap_names_all)):
            # print("image_name:",image_name)
            # print("Gaumap_name:",gaumap_name)
            if os.path.isfile(image_name) and os.path.isfile(gaumap_name) and os.path.isfile(gaumap_name_all):
                gaumap = cv2.imread(gaumap_name, 0)
                gaumaps.append(gaumap)

                gaumap_all = cv2.imread(gaumap_name_all, 0)
                gaumaps_all.append(gaumap_all)
                # print("gaumap:",gaumap[:,:,np.newaxis].shape)
                # print("images[n]",images[n].shape)
                # print("concat", np.concatenate((images[n], gaumap[:,:,np.newaxis]), axis=2).shape)
                images_gaumaps.append(np.concatenate((images[n], gaumap[:,:,np.newaxis], gaumap_all[:,:,np.newaxis]), axis=2)) 
        
        images = images_gaumaps

        # print("images:",np.array(images).shape) #(2239,256,256,5)
        # print("labels:",np.array(labels).shape) #(2239, 4,256,256)
        ######load images and gaussian maps for test
        gaumaps_test = []
        gaumaps_test_all = []
        images_gaumaps_test = []
        for n, (image_name_test, gaumap_name_test, gaumap_name_test_all) in enumerate(zip(image_names_test, gaumap_names_test, gaumap_names_test_all)):
            # print("image_name:",image_name_test)
            # print("Gaumap_name:",gaumap_name_test)
            if os.path.isfile(image_name_test) and os.path.isfile(gaumap_name_test) and os.path.isfile(gaumap_name_test_all):
                test_gaumap = cv2.imread(gaumap_name_test, 0)
                gaumaps_test.append(test_gaumap)

                test_gaumap_all = cv2.imread(gaumap_name_test_all, 0) 
                gaumaps_test_all.append(test_gaumap_all)
                # print("gaumap:",gaumap[:,:,np.newaxis].shape)
                # print("images[n]",images[n].shape)
                # print("concat", np.concatenate((images[n], gaumap[:,:,np.newaxis]), axis=2).shape)
                images_gaumaps_test.append(np.concatenate((test_images[n], test_gaumap[:,:,np.newaxis], test_gaumap_all[:,:,np.newaxis]), axis=2)) 
        
        test_images = images_gaumaps_test

        # print("test_images:",np.array(test_images).shape) #(560, 256, 256, 5)
        # print("test_labels:",np.array(test_labels).shape) #(560, 4, 256, 256)
        #print("image_names_test:",image_names_test[::200])

        #print("args.all_channels:",args.all_channels) #True
        # training with all channels
        if args.all_channels:
            img = images[0]
            if img.ndim==3:
                nchan = 4
            elif img.ndim==2:
                nchan = 1
            channels = None 
        else:
            nchan = 2 

        #print("nchan:",nchan) # 4
        
        # model path
        #print("args.diam_mean:",args.diam_mean)
        # print("model_type:",model_type) #None
        # print("pretrained_model:",pretrained_model) # False
        # print("os.path.exists(pretrained_model)2:",os.path.exists(pretrained_model)) #True
        #print("os.path.exists(pretrained_model)2:",os.path.exists(True))
        

        logger.info('>>>> during training rescaling images to fixed diameter of %0.1f pixels'%args.diam_mean)
            
        # initialize model
        if args.unet:
            model = core.UnetModel(device=device,
                                    pretrained_model=pretrained_model, 
                                    diam_mean=szmean,
                                    residual_on=args.residual_on,
                                    style_on=args.style_on,
                                    concatenation=args.concatenation,
                                    nclasses=args.nclasses,
                                    nchan=nchan)
        else:
            # print("args.residual_on:",args.residual_on) #1
            # print("args.concatenation:",args.concatenation)#0
            # print("nchan:",nchan) # 3
            # print("args.style_on:",args.style_on) #1
            #print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            model = models.CellposeModel(device=device,
                                        pretrained_model=pretrained_model if model_type is None else None,
                                        model_type=model_type, 
                                        diam_mean=szmean,
                                        residual_on=args.residual_on,
                                        style_on=args.style_on,
                                        concatenation=args.concatenation,
                                        nchan=nchan)
        
        # train segmentation model
        if args.train:
            # print("channels:",channels) #None
            # print("os.path.realpath(args.dir):",os.path.realpath(args.dir)) # /hpc/group/jilab/imaging/Yuxing/cellpose/train_pciSeq
            # print("args.save_every;",args.save_every) #100
            # print("args.save_each:",args.save_each) #False
            #print("args.min_train_masks:",args.min_train_masks) #5
            #print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

            cpmodel_path = model.train(images, labels, train_files=image_names,
                                        test_data=test_images, test_labels=test_labels, test_files=image_names_test,
                                        learning_rate=args.learning_rate, 
                                        weight_decay=args.weight_decay,
                                        channels=channels,
                                        save_path=os.path.realpath(args.dir), save_every=args.save_every,
                                        save_each=args.save_each,
                                        n_epochs=args.n_epochs,
                                        batch_size=args.batch_size, 
                                        min_train_masks=args.min_train_masks)
            model.pretrained_model = cpmodel_path
            logger.info('>>>> model trained and saved to %s'%cpmodel_path)

        # train size model
        if args.train_size:
            sz_model = models.SizeModel(cp_model=model, device=device)
            masks = [lbl[0] for lbl in labels]
            test_masks = [lbl[0] for lbl in test_labels] if test_labels is not None else test_labels
            # data has already been normalized and reshaped
            sz_model.train(images, masks, test_images, test_masks, 
                            channels=None, normalize=False,
                                batch_size=args.batch_size)
            if test_images is not None:
                predicted_diams, diams_style = sz_model.eval(test_images, 
                                                                channels=None,
                                                                normalize=False)
                ccs = np.corrcoef(diams_style, np.array([utils.diameters(lbl)[0] for lbl in test_masks]))[0,1]
                cc = np.corrcoef(predicted_diams, np.array([utils.diameters(lbl)[0] for lbl in test_masks]))[0,1]
                logger.info('style test correlation: %0.4f; final test correlation: %0.4f'%(ccs,cc))
                np.save(os.path.join(args.test_dir, '%s_predicted_diams.npy'%os.path.split(cpmodel_path)[1]), 
                        {'predicted_diams': predicted_diams, 'diams_style': diams_style})

if __name__ == '__main__':
    main()
    
