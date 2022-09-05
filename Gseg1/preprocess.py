import torch
import torch.nn.functional as F
from monai.data.utils import dense_patch_slices
import matplotlib.pyplot as plt
import cv2
import glob
import csv
from natsort import natsorted
import os
import sys
import h5py
import numpy as np
import utils, metrics, models, io_img, core, plot, dynamics
from io_img import logger_setup
import scipy.io as io
from PIL import Image
import fastremap

def filter_spots(label, genes_spot):
    if len(genes_spot) != 0:
        filted_spot = []
        xs = genes_spot[:, 0]
        ys = genes_spot[:, 1]
        gene_mask = label[ys,xs]!=0
        filtered_genes = genes_spot[gene_mask]
    else:
        assert len(genes_spot)!= 0, "The number of genes is 0"
    
    return filtered_genes

def gen_pose_target(joints, device, h=256, w=256, sigma=7):
    #print "Target generation -- Gaussian maps"
    '''
    joints : gene spots #[N,2]
    sigma : 7
    '''
    
    joint_num = joints.shape[0] #16

    gaussian_maps = torch.zeros((joint_num, h, w)).to(device)

    for ji in range(0, joint_num):
        gaussian_maps[ji, :, :] = gen_single_gaussian_map(joints[ji, :], h, w, sigma, device)

    # Get background heatmap
    max_heatmap = torch.max(gaussian_maps, 0).values #cuda
    return max_heatmap

def gen_single_gaussian_map(center, h, w, sigma, device):
    #print "Target generation -- Single gaussian maps"
    '''
    center a gene spot #[2,]
    '''

    grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    inds = torch.stack([grid_x,grid_y], dim=0).to(device)
    d2 = (inds[0] - center[0]) * (inds[0] - center[0]) + (inds[1] - center[1]) * (inds[1] - center[1]) #[256,256]
    exponent = d2 / 2.0 / sigma / sigma
    exp_mask = exponent > 4.6052
    exponent[exp_mask] = 0
    gaussian_map = torch.exp(-exponent)
    gaussian_map[exp_mask] = 0
    gaussian_map[gaussian_map>1] = 1

    return gaussian_map

def sliding_window_inference(inputs, label, spot, spot_in_cell, roi_size, sw_batch_size, sigma, save_dir, device):
    """Use SlidingWindow method to execute inference.

    Args:
        inputs (torch Tensor): input image to be processed (assuming NCHW)
        roi_size (list, tuple): the window size to execute SlidingWindow inference.
        sw_batch_size (int): the batch size to run window slices.

    Note:
        must be channel first, support 2D.
        input data must have batch dim.
        execute on 1 image/per inference, run a batch of window slices of 1 input image.
    """
    
    num_spatial_dims = len(inputs.shape) - 2 # 2
    assert len(roi_size) == num_spatial_dims, 'roi_size {} does not match input dims.'.format(roi_size)

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    image_size = list(inputs.shape[2:]) # image [h, w] [5548, 6130]
    batch_size = inputs.shape[0] #1

    # TODO: Enable batch sizes > 1 in future
    if batch_size > 1:
        raise NotImplementedError

    original_image_size = [image_size[i] for i in range(num_spatial_dims)]
    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = [i for k in range(len(inputs.shape) - 1, 1, -1) for i in (0, max(roi_size[k - 2] - inputs.shape[k], 0))]
    inputs = F.pad(inputs, pad=pad_size, mode='constant', value=0)
    label = F.pad(label, pad=pad_size, mode='constant', value=0)

    # TODO: interval from user's specification
    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims)
    
    # Store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    print("The number of cropped image is", len(slices))
    
    input_slices = []
    label_slices = []
    spot_slices = []
    gaussian_slices = []
    gaussian_all_slices = []
    for slice_index in range(0, len(slices), sw_batch_size):
        slice_index_range = range(slice_index, min(slice_index + sw_batch_size, len(slices)))
        for curr_index in slice_index_range:
            slice_i, slice_j = slices[curr_index]
            
            im = inputs[0, :, slice_i, slice_j]
            lb = label[0, 0, slice_i, slice_j]

            #generate Gaussian Map
            y_min = slice_i.start
            y_max = slice_i.stop
            x_min = slice_j.start
            x_max = slice_j.stop

            x_min_mask = spot[:, 0] >= x_min
            x_max_mask = spot[:, 0] < x_max
            y_min_mask = spot[:, 1] >= y_min
            y_max_mask = spot[:, 1] < y_max
            spot_mask = x_min_mask*x_max_mask*y_min_mask*y_max_mask
            crop_spots = spot[spot_mask]-torch.tensor([[x_min,y_min]]).to(device)
            
            x_min_in_cell_mask = spot_in_cell[:, 0] >= x_min
            x_max_in_cell_mask = spot_in_cell[:, 0] < x_max
            y_min_in_cell_mask = spot_in_cell[:, 1] >= y_min
            y_max_in_cell_mask = spot_in_cell[:, 1] < y_max
            spot_mask = x_min_in_cell_mask*x_max_in_cell_mask*y_min_in_cell_mask*y_max_in_cell_mask
            crop_spots_in_cell = spot_in_cell[spot_mask]-torch.tensor([[x_min,y_min]]).to(device)
            
            gaumap_all = torch.zeros(roi_size[0], roi_size[1]).to(device)
            if len(crop_spots) != 0:
                gaumap_all = gen_pose_target(crop_spots, device, roi_size[0], roi_size[1], sigma)
            
            gaumap = torch.zeros(roi_size[0], roi_size[1]).to(device)
            if len(crop_spots_in_cell) != 0:
                gaumap = gen_pose_target(crop_spots_in_cell, device, roi_size[0], roi_size[1], sigma)

            cv2.imwrite("{}/images/{}_image.jpg".format(save_dir, curr_index), im.permute((1,2,0)).cpu().numpy())
            
            lb_Image = Image.fromarray((fastremap.renumber(lb.cpu().numpy(), in_place=True)[0]).astype(np.uint16))
            lb_path = os.path.join("{}/labels/{}_label.png".format(save_dir, curr_index))
            lb_Image.save(lb_path)

            with open('{}/spots/{}_spot.csv'.format(save_dir, curr_index), 'w') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['spotX','spotY'])
                writer.writerows(crop_spots.cpu().numpy())
            
            cv2.imwrite("{}/GauMaps/GauMap/{}_gaumap.jpg".format(save_dir, curr_index), gaumap.cpu().numpy()*255)
            cv2.imwrite("{}/GauMaps/GauMap_all/{}_gaumap_all.jpg".format(save_dir, curr_index), gaumap_all.cpu().numpy()*255)

            input_slices.append(im)    
            label_slices.append(lb)
            spot_slices.append(crop_spots)
            gaussian_slices.append(gaumap)
            gaussian_all_slices.append(gaumap_all)

    C_images = torch.stack(input_slices)
    C_labels = torch.stack(label_slices).unsqueeze(1)
    C_gaussian = torch.stack(gaussian_slices).unsqueeze(1)
    C_gaussian_all = torch.stack(gaussian_all_slices).unsqueeze(1)
    C_spots = spot_slices
    
    print("C_images:", C_images.shape)
    print("C_labels:", C_labels.shape)
    print("C_gaussian:", C_gaussian.shape)
    print("C_gaussian_all:", C_gaussian_all.shape)
    print("C_spots:", len(C_spots))

    return C_images, C_labels, C_spots, C_gaussian, C_gaussian_all

def _get_scan_interval(image_size, roi_size, num_spatial_dims):
    assert (len(image_size) == num_spatial_dims), 'image coord different from spatial dims.'
    assert (len(roi_size) == num_spatial_dims), 'roi coord different from spatial dims.'

    scan_interval = [1 for _ in range(num_spatial_dims)] #[1,1]
    for i in range(num_spatial_dims):
        if roi_size[i] == image_size[i]: # image_size:[5548,6130] 
            scan_interval[i] = int(roi_size[i])
        else:
            # this means that it's r-16 (if r>=64) and r*0.75 (if r<=64)
            scan_interval[i] = int(max(roi_size[i] - 16, roi_size[i] * 0.75))
    return tuple(scan_interval) #(240,240)

def preprocessing(image, label, spot, sigma, roi_size, sw_batch_size, save_dir, device='cpu'):
    saved_image_path = os.path.join(save_dir, 'images')
    print("saved_image_path:", saved_image_path)
    if not os.path.exists(saved_image_path):
        os.makedirs(saved_image_path)

    saved_label_path = os.path.join(save_dir, 'labels')
    print("saved_label_path:", saved_label_path)
    if not os.path.exists(saved_label_path):
        os.makedirs(saved_label_path)

    saved_spot_path = os.path.join(save_dir, 'spots')
    print("saved_spot_path:", saved_spot_path)
    if not os.path.exists(saved_spot_path):
        os.makedirs(saved_spot_path)

    saved_gaumap_path = os.path.join(save_dir, 'GauMaps')
    print("saved_gaumap_path:", saved_gaumap_path)
    if not os.path.exists(saved_gaumap_path):
        os.makedirs(saved_gaumap_path)
    
    gaumap_path = os.path.join(saved_gaumap_path, 'GauMap')
    gaumap_all_path = os.path.join(saved_gaumap_path, 'GauMap_all')
    if not os.path.exists(gaumap_path):
        os.makedirs(gaumap_path)
    if not os.path.exists(gaumap_all_path):
        os.makedirs(gaumap_all_path)
    
    label = label.astype(np.int32)
    spot = spot[:,1:].astype(np.float64).astype(np.int32)
    spot_in_cell = filter_spots(label, spot)
    
    #convert numpy data to tensor
    image = torch.from_numpy(image).permute((2,0,1)).unsqueeze(0).to(device)
    label = torch.from_numpy(label).unsqueeze(0).unsqueeze(0).to(device)
    spot = torch.from_numpy(spot).to(device)
    spot_in_cell = torch.from_numpy(spot_in_cell).to(device)

    C_images, C_labels, C_spots, C_gaussian, C_gaussian_all = sliding_window_inference(image, label, spot, spot_in_cell, roi_size, sw_batch_size, sigma, save_dir, device)
    
    return C_images, C_labels, C_spots, C_gaussian, C_gaussian_all


