import torch
import torch.nn.functional as F
from monai.data.utils import dense_patch_slices
import matplotlib.pyplot as plt
import cv2
import glob
from natsort import natsorted
import os
import sys
import h5py
import numpy as np
import utils, metrics, models, Gseg_io, core, plot, dynamics
from Gseg_io import logger_setup
import scipy.io as io
from PIL import Image

def filter_spots(label, genes_spot):
    total_spot_num = len(genes_spot)

    filted_spot = []
    if total_spot_num != 0:
        for x,y in genes_spot:
            if label[y,x]!=0:
                filted_spot.append([x,y])
    
    return np.array(filted_spot)

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

def sliding_window_inference(inputs, roi_size, sw_batch_size, predictor, spot, sigma, filename, save_dir, device='cpu'):
    """Use SlidingWindow method to execute inference.

    Args:
        inputs (torch Tensor): input image to be processed (assuming NCHW[D])
        roi_size (list, tuple): the window size to execute SlidingWindow inference.
        sw_batch_size (int): the batch size to run window slices.
        predictor (Callable): given input tensor `patch_data` in shape NCHW[D], `predictor(patch_data)`
            should return a prediction with the same spatial shape and batch_size, i.e. NMHW[D];
            where HW[D] represents the patch spatial size, M is the number of output channels, N is `sw_batch_size`.

    Note:
        must be channel first, support both 2D and 3D.
        input data must have batch dim.
        execute on 1 image/per inference, run a batch of window slices of 1 input image.
    """
    '''
        inputs: [1, 3, h, w]
        roi_size: 256
        sw_batch_size: 1
        spot: (N,2)
        sigma: 7
    '''
    
    num_spatial_dims = len(inputs.shape) - 2 # 2
    assert len(roi_size) == num_spatial_dims, 'roi_size {} does not match input dims.'.format(roi_size)

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    image_size = list(inputs.shape[2:]) # image [h, w] [5548, 6130]
    batch_size = inputs.shape[0] #1

    # TODO: Enable batch sizes > 1 in future
    if batch_size > 1:
        raise NotImplementedError

    original_image_size = [image_size[i] for i in range(num_spatial_dims)] # [5548, 6130]
    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = [i for k in range(len(inputs.shape) - 1, 1, -1) for i in (0, max(roi_size[k - 2] - inputs.shape[k], 0))] #[0,0,0,0]
    inputs = F.pad(inputs, pad=pad_size, mode='constant', value=0) # same with initial inputs [1, 3, 5548, 6130] 

    # TODO: interval from user's specification
    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims) #(240,240)
    
    # Store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval) #(624, 2)

    saved_gaumap_path = os.path.join(os.path.join(save_dir, 'GauMap'), filename)
    if not os.path.exists(saved_gaumap_path):
        os.makedirs(saved_gaumap_path)
    
    slice_batches = []
    for slice_index in range(0, len(slices), sw_batch_size):
        slice_index_range = range(slice_index, min(slice_index + sw_batch_size, len(slices)))
        input_slices = []
        for curr_index in slice_index_range:
            if num_spatial_dims == 3:
                slice_i, slice_j, slice_k = slices[curr_index]
                input_slices.append(inputs[0, :, slice_i, slice_j, slice_k])
            else:
                slice_i, slice_j = slices[curr_index]
                im = inputs[0, :, slice_i, slice_j]

                if not os.path.exists("{}/{}_gaumap.jpg".format(saved_gaumap_path,curr_index)):
                    #generate Gaussian Map
                    y_min = slice_i.start
                    y_max = slice_i.stop
                    x_min = slice_j.start
                    x_max = slice_j.stop

                    x_min_in_cell_mask = spot[:, 0] >= x_min
                    x_max_in_cell_mask = spot[:, 0] < x_max
                    y_min_in_cell_mask = spot[:, 1] >= y_min
                    y_max_in_cell_mask = spot[:, 1] < y_max
                    spot_mask = x_min_in_cell_mask*x_max_in_cell_mask*y_min_in_cell_mask*y_max_in_cell_mask
                    crop_spots = spot[spot_mask]-torch.tensor([[x_min,y_min]]).to(device)

                    gaumap = torch.zeros(256, 256).to(device)
                    if len(crop_spots) != 0:
                        gaumap = gen_pose_target(crop_spots, device, 256, 256, sigma)

                    cv2.imwrite("{}/{}_gaumap.jpg".format(saved_gaumap_path,curr_index), gaumap.cpu().numpy()*255)
                    gaumap = gaumap.unsqueeze(0)*255
                
                else:
                    print("Existed gaumap {}_gaumap.jpg".format(curr_index))
                    gaumap = cv2.imread("{}/{}_gaumap.jpg".format(saved_gaumap_path,curr_index) ,0)[np.newaxis, :, :]
                    gaumap = torch.from_numpy(gaumap).cuda()

                image_gaumap = torch.cat([im, gaumap], dim=0)
                input_slices.append(image_gaumap) 

        slice_batches.append(torch.stack(input_slices))

    # Perform predictions
    output_rois = list()
    output_offsetmap = list()
    output_confidencemap = list()
    output_centermap = list()

    for data in slice_batches:
        data = data.permute((0,2,3,1)).cpu().numpy()
        
        offsetmap_list = []
        confidencemap_list = []
        centermap_list = []
        for each_input in data:
            seg_prob = predictor.eval(each_input, channels=None, diameter=32.0,
                                    do_3D=False, net_avg=True,
                                    augment=False,
                                    resample=True,
                                    flow_threshold=0.4,
                                    cellprob_threshold=0.5,
                                    stitch_threshold=0.0,
                                    invert=False,
                                    batch_size=8,
                                    interp=True,
                                    normalize=True,
                                    channel_axis=None,
                                    z_axis=None,
                                    anisotropy=1.0,
                                    model_loaded=True)  # batched patch segmentation
            masks, flows = seg_prob[:2]

            offsetmap_list.append(torch.from_numpy(flows[1]).to(device))
            confidencemap_list.append(torch.from_numpy(flows[2]).unsqueeze(0).to(device))
            centermap_list.append(torch.from_numpy(flows[3]).unsqueeze(0).to(device))
        
        output_offsetmap.append(torch.stack(offsetmap_list))
        output_confidencemap.append(torch.stack(confidencemap_list))
        output_centermap.append(torch.stack(centermap_list))
    
    # stitching output image
    output_classes = output_confidencemap[0].shape[1]
    output_shape1 = [batch_size, output_classes] + list(image_size)

    output_classes = output_offsetmap[0].shape[1]
    output_shape2 = [batch_size, output_classes] + list(image_size)

    output_classes = output_centermap[0].shape[1]
    output_shape3 = [batch_size, output_classes] + list(image_size)

    # allocate memory to store the full output and the count for overlapping parts
    output_confimap = torch.zeros(output_shape1, dtype=torch.float32, device=inputs.device)
    output_offmap = torch.zeros(output_shape2, dtype=torch.float32, device=inputs.device)
    output_cenmap = torch.zeros(output_shape3, dtype=torch.float32, device=inputs.device)
    count_map = torch.zeros(output_shape1, dtype=torch.float32, device=inputs.device)

    for window_id, slice_index in enumerate(range(0, len(slices), sw_batch_size)):
        slice_index_range = range(slice_index, min(slice_index + sw_batch_size, len(slices)))
        # store the result in the proper location of the full output
        for curr_index in slice_index_range:
            if num_spatial_dims == 3:
                slice_i, slice_j, slice_k = slices[curr_index]
                output_image[0, :, slice_i, slice_j, slice_k] += output_rois[window_id][curr_index - slice_index, :]
                count_map[0, :, slice_i, slice_j, slice_k] += 1.
            else:
                slice_i, slice_j = slices[curr_index]
                output_confimap[0, :, slice_i, slice_j] += output_confidencemap[window_id][curr_index - slice_index, :]
                output_offmap[0, :, slice_i, slice_j] += output_offsetmap[window_id][curr_index - slice_index, :]
                output_cenmap[0, :, slice_i, slice_j] += output_centermap[window_id][curr_index - slice_index, :]
                count_map[0, :, slice_i, slice_j] += 1.

    # account for any overlapping sections
    output_confimap /= count_map
    output_offmap /= count_map
    output_cenmap /= count_map

    if num_spatial_dims == 3:
        return output_image[..., :original_image_size[0], :original_image_size[1], :original_image_size[2]]

    return output_confimap[..., :original_image_size[0], :original_image_size[1]], output_offmap[..., :original_image_size[0], :original_image_size[1]], output_cenmap[..., :original_image_size[0], :original_image_size[1]] # 2D

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

def load_data_and_model(root_dir, save_dir, sigma):
    dapi_image_file = natsorted(glob.glob(os.path.join(os.path.join(root_dir, 'image/'), '*.jpg')))
    dapi_mask_file = natsorted(glob.glob(os.path.join(os.path.join(root_dir, 'mask/'), '*.jpg')))
    dapi_spots_file = natsorted(glob.glob(os.path.join(os.path.join(root_dir, 'spots/'), '*.csv')))

    #load your pre-trained model
    model_file = os.path.join(root_dir, "model path")
    
    print("dapi_image_file:", len(dapi_image_file))
    print("dapi_mask_file:", len(dapi_mask_file))
    print("dapi_spots_file:", len(dapi_spots_file))
    print("model_file:", model_file)

    assert len(dapi_image_file) == len(dapi_mask_file) == len(dapi_spots_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    if torch.cuda.is_available():
        print("use gpu")
        gpu = True
    else:
        print("use cpu")
        gpu = False
    
    pretrained_model = model_file
    model_type = None
    szmean = 34.0
    residual_on = 1
    style_on = 1
    concatenation = 0
    nchan = 2

    model = models.GeneSegModel(gpu=gpu, device=device, 
                                pretrained_model=pretrained_model,
                                model_type=model_type,
                                diam_mean=szmean,
                                residual_on=residual_on,
                                style_on=style_on,
                                concatenation=concatenation,
                                net_avg=False,
                                nchan=nchan)
    
    for i, (image_name, spot_name, mask_name) in enumerate(zip(dapi_image_file, dapi_spots_file, dapi_mask_file)):
        print("image_name:", image_name)
        print("mask_name:", mask_name)
        print("spot_name:", spot_name)
        filename = os.path.basename(image_name)[5:-4]
        print("filename:", filename)

        #load image
        image = cv2.imread(image_name)
        image = image[:,:,0]

        #load spots
        with open(spot_name, 'r') as f:
            lines = f.readlines()
            lines = lines[1:]

        spot_list = []
        for line in lines:
            splits = line.split(',')
            spot_list.append([int(float(splits[1])), int(float(splits[2]))])
        
        spot = np.array(spot_list)

        #convert numpy image to tensor image
        image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)
        spot = torch.from_numpy(spot).to(device)

        confidence, offset, center = sliding_window_inference(image, [256, 256], 10, model, spot, sigma, filename, save_dir, device)
        outputs = dynamics.compute_masks(offset.squeeze().cpu().numpy(), center.squeeze().cpu().numpy(), confidence.squeeze().cpu().numpy(), confidence_threshold=0.8,
                                                         flow_threshold=0.4, interp=True, resize=None,
                                                         use_gpu=True, device='cuda:0')
        
        
        wholemask = outputs
        kernel = np.ones((5,5), np.uint8)
        wholemask = cv2.morphologyEx(wholemask, cv2.MORPH_OPEN, kernel)

        plt.imshow(wholemask)
        plt.axis("off")
        plt.savefig("{}/Ours_label/CellMap_{}_image.jpg".format(save_dir, filename), bbox_inches='tight', pad_inches = 0)
        plt.clf()

        im = Image.fromarray(wholemask)
        im_path = os.path.join(os.path.join(os.path.join(save_dir, 'Ours_label'),'CellMap_{}.png'.format(filename)))
        im.save(im_path)

        io.savemat(os.path.join(os.path.join(save_dir, 'Ours_label'),'CellMap_{}.mat'.format(filename)), {'CellMap': wholemask})
    

if __name__ == '__main__':
    # load inputs
    root_dir = 'your raw image path'
    save_dir = 'save path'
    sigma = 7

    logger, log_file = logger_setup()
    load_data_and_model(root_dir, save_dir, sigma)

