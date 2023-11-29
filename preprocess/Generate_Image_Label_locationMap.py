import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import fastremap
from natsort import natsorted
from scipy.ndimage import find_objects, gaussian_filter, generate_binary_structure, label, maximum_filter1d, binary_fill_holes
from scipy.spatial import ConvexHull
import random
import csv
import torch

def masks_to_outlines(masks):
    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError('masks_to_outlines takes 2D or 3D array, not %dD array'%masks.ndim)
    outlines = np.zeros(masks.shape, bool)
    
    if masks.ndim==3:
        for i in range(masks.shape[0]):
            outlines[i] = masks_to_outlines(masks[i])
        return outlines
    else:
        slices = find_objects(masks.astype(int))
        for i,si in enumerate(slices):
            if si is not None:
                sr,sc = si
                mask = (masks[sr, sc] == (i+1)).astype(np.uint8)
                contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                pvc, pvr = np.concatenate(contours[-2], axis=0).squeeze().T            
                vr, vc = pvr + sr.start, pvc + sc.start 
                outlines[vr, vc] = 1
        return outlines

def filter_spots(label, genes_spot):
    if len(genes_spot) != 0:
        filted_spot = []
        xs = genes_spot[:, 0].astype(np.int64)
        ys = genes_spot[:, 1].astype(np.int64)
        xsmask = xs<label.shape[1]
        ysmask = ys<label.shape[0]
        mask1 = xsmask*ysmask

        xsmask = xs>0
        ysmask = ys>0
        mask2 = xsmask*ysmask
        mask = mask1*mask2

        xs = xs[mask]
        ys = ys[mask]
        genes_spot = genes_spot[mask]
        gene_mask = label[ys,xs]!=0
        filtered_genes = genes_spot[gene_mask]
    else:
        # print("The number of genes is 0")
        filtered_genes = np.array([])
    
    return filtered_genes

def custom_blur_demo(image):
    kernel = np.array([[0,-1,0],[-1,8,-1],[0,-1,0]], np.float32)
    dst = cv2.filter2D(image, -1, kernel=kernel)
    #cv2.imshow("custom_blur_demo", dst)
    return dst

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

def main(root_image_dir, root_seg_dir, root_spot_dir, save_dir):
    image_files = natsorted(glob.glob(os.path.join(root_image_dir, '*.jpg'))) #Modify '.jpg' according to the suffix name of your data
    label_files = natsorted(glob.glob(os.path.join(root_seg_dir, '*.png')))
    spot_files = natsorted(glob.glob(os.path.join(root_spot_dir,'*.csv')))

    print("the number of images, labels, and spots:", len(image_files), len(label_files), len(spot_files))
    assert len(image_files) == len(label_files) == len(spot_files)

    for index, (imagefile, labelfile, spot_file) in enumerate(zip(image_files, label_files, spot_files)):
        foldername = os.path.splitext(os.path.basename(imagefile))[0]
        print(foldername)
        save_subdir = os.path.join(save_dir, foldername)
        print("save_subdir:", save_subdir)

        saved_image_path = os.path.join(save_subdir, 'images')
        print("saved_image_path:", saved_image_path)
        if not os.path.exists(saved_image_path):
            os.makedirs(saved_image_path)

        saved_label_path = os.path.join(save_subdir, 'labels')
        print("saved_label_path:", saved_label_path)
        if not os.path.exists(saved_label_path):
            os.makedirs(saved_label_path)

        saved_spot_path = os.path.join(save_subdir, 'spots')
        print("saved_spot_path:", saved_spot_path)
        if not os.path.exists(saved_spot_path):
            os.makedirs(saved_spot_path)

        saved_gaumap_path = os.path.join(save_subdir, 'HeatMaps')
        print("saved_gaumap_path:", saved_gaumap_path)
        if not os.path.exists(saved_gaumap_path):
            os.makedirs(saved_gaumap_path)
        
        gaumap_path = os.path.join(saved_gaumap_path, 'HeatMap')
        gaumap_all_path = os.path.join(saved_gaumap_path, 'HeatMap_all')
        if not os.path.exists(gaumap_path):
            os.makedirs(gaumap_path)
        if not os.path.exists(gaumap_all_path):
            os.makedirs(gaumap_all_path)

        imagefiles = [imagefile]*100
        labelfiles = [labelfile]*100
        spotfiles = [spot_file]*100

        assert len(imagefiles) == len(labelfiles) == len(spotfiles)

        random.seed(1)
        for index, (imagefile, labelfile, spot_file) in enumerate(zip(imagefiles, labelfiles, spotfiles)):
            filename = os.path.basename(imagefile)[:-4]
            labelname = os.path.basename(labelfile)[:-4]
            spotsname = os.path.basename(spot_file)[:-4]

            with open(spot_file, 'r') as f:
                lines = f.readlines()
                lines = lines[1:]

            spot_list = []
            for line in lines:
                splits = line.strip().split(',')
                spot_list.append([int(float(splits[0])), int(float(splits[1])), splits[2]])
                
            spot = np.array(spot_list)
            
            im = cv2.imread(imagefile)
            label = cv2.imread(labelfile, -1)

            img_h, img_w, _ = im.shape
            h_off = random.randint(0, img_h - 256)
            w_off = random.randint(0, img_w - 256)
            
            im = im[h_off: h_off + 256, w_off: w_off + 256]
            label = label[h_off: h_off + 256, w_off: w_off + 256]

            minx = w_off
            maxx = w_off + 256
            miny = h_off
            maxy = h_off + 256

            crop_spot_list = []
            for x, y, gene in spot:
                x = int(float(x))
                y = int(float(y))
                if x >= minx and x < maxx and y >= miny and y < maxy:
                    crop_spot_list.append([x-minx, y-miny, gene])

            spot = np.array(crop_spot_list)
            spot_in_cell = filter_spots(label, spot)

            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            gaumap_all = torch.zeros(label.shape[0], label.shape[1]).to(device)
            if len(spot) != 0:
                spot_tensor = torch.from_numpy(spot[:, :2].astype(np.int64)).to(device)
                gaumap_all = gen_pose_target(spot_tensor, device, label.shape[0], label.shape[1], 7)

            gaumap = torch.zeros(label.shape[0], label.shape[1]).to(device)
            if len(spot_in_cell) != 0:    
                spot_in_cell_tensor = torch.from_numpy(spot_in_cell[:, :2].astype(np.int64)).to(device)
                gaumap = gen_pose_target(spot_in_cell_tensor, device, label.shape[0], label.shape[1], 7)
            
            cv2.imwrite("{}/{}_gaumap_all.jpg".format(gaumap_all_path, index), gaumap_all.cpu().numpy()*255)
            cv2.imwrite("{}/{}_gaumap.jpg".format(gaumap_path, index), gaumap.cpu().numpy()*255)

            cv2.imwrite("{}/{}_image.jpg".format(saved_image_path, index), im)

            label = Image.fromarray(label)
            lbl_path = "{}/{}_label.png".format(saved_label_path,index)
            label.save(lbl_path)

            with open( '{}/{}_spot.csv'.format(saved_spot_path, index), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['spotX','spotY', 'gene'])
                writer.writerows(spot)

if __name__ == '__main__':

    base_dir = "add your data path"
    root_image_dir = "{}/images/".format(base_dir)
    root_seg_dir = "{}/labels/".format(base_dir)
    root_spot_dir = "{}/spots/".format(base_dir)
    save_crop_dir = "add your path to save results"

    main(root_image_dir, root_seg_dir, root_spot_dir, save_crop_dir)

