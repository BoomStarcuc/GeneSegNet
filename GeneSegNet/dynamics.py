import time, os
from scipy.ndimage.filters import maximum_filter1d
import torch
import scipy.ndimage
import numpy as np
import tifffile
from tqdm import trange
import matplotlib.pyplot as plt
from numba import njit, float32, int32, vectorize
import cv2
import fastremap
from scipy.ndimage.filters import gaussian_filter
import scipy.cluster.hierarchy as hcluster
from numpy.core.records import fromarrays
import math
import kornia
from morphology import Dilation2d, Erosion2d

import logging
dynamics_logger = logging.getLogger(__name__)

import utils, metrics, transforms, plot
from torch import optim, nn
import resnet_torch
TORCH_ENABLED = True 
torch_GPU = torch.device('cuda')
torch_CPU = torch.device('cpu')

def mat_math (intput, str):
    if str=="atan":
        output = torch.atan(intput) 
    if str=="sqrt":
        output = torch.sqrt(intput) 
    return output

def level_set(LSF, img, mu, nu, epison, step):
    Drc = (epison / math.pi) / (epison*epison+ LSF*LSF)
    Hea = 0.5*(1 + (2 / math.pi) * mat_math(LSF/epison, "atan")) 
    Iys = torch.gradient(LSF, dim=2)[0]
    Ixs = torch.gradient(LSF, dim=3)[0]
    s = mat_math(Ixs*Ixs+Iys*Iys, "sqrt") 
    Nx = Ixs / (s+0.000001) 
    Ny = Iys / (s+0.000001)

    Mxx = torch.gradient(Nx, dim=2)[0]
    Nxx = torch.gradient(Nx, dim=3)[0]

    Nyy = torch.gradient(Ny, dim=2)[0]
    Myy = torch.gradient(Ny, dim=3)[0]

    cur = Nxx + Nyy
    Length = nu*Drc*cur 
    
    Lap = kornia.filters.laplacian(LSF, 3)
    Penalty = mu*(Lap - cur) 

    s1=Hea*img 
    s2=(1-Hea)*img 
    s3=1-Hea 
    C1 = s1.sum()/ Hea.sum() 
    C2 = s2.sum()/ s3.sum() 
    CVterm = Drc*(-1 * (img - C1)*(img - C1) + 1 * (img - C2)*(img - C2)) 

    LSF = LSF + step*(Length + Penalty + CVterm) 
    return LSF 


def postprocess(mask, N, device='cpu'):
    if N == 1:
        dilation = Dilation2d(1, 1, 5, soft_max=False).to(device)
        erosion = Erosion2d(1, 1, 5, soft_max=True).to(device)

    mask = torch.from_numpy(mask.astype(np.int32)).unsqueeze(0).unsqueeze(0).to(device)
    cell_ids = torch.unique(mask)[1:]
    
    mu = 1 
    nu = 0.003 * 255 * 255 
    num = 150
    epison = 1 
    step = 0.01

    new_mask = torch.zeros((mask.shape[2],mask.shape[3]), dtype=torch.int32).to(device)
    for cell_id in cell_ids:
        img = ((mask == cell_id)*255).float()
        LSF = ((mask == cell_id)*1).float()

        for i in range(1,num):
            if N==1 and i == 1:
                img = erosion(img)
                img = dilation(img)
                img = dilation(img)
            
            LSF = level_set(LSF, img, mu, nu, epison, step)

        LSF[:][LSF[:] >= 0] = 1
        LSF[:][LSF[:] < 0] = 0

        outcoord = torch.nonzero(LSF.squeeze())
        new_mask[outcoord[:,0], outcoord[:,1]] = cell_id

    new_mask = new_mask.detach().squeeze().cpu().numpy()
    return new_mask

def gen_pose_target(joints, device, h=256, w=256, sigma=3):
    #print "Target generation -- Gaussian maps"
    if joints.shape[0]!=0:
        joint_num = joints.shape[0]
        gaussian_maps = torch.zeros((joint_num, h, w)).to(device)

        for ji in range(0, joint_num):
            gaussian_maps[ji, :, :] = gen_single_gaussian_map(joints[ji, :], h, w, sigma, device)

        # Get background heatmap
        max_heatmap = torch.max(gaussian_maps, 0).values
    else:
        max_heatmap = torch.zeros((h, w)).to(device)
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

def masks_to_flows_gpu(masks, device=None):
    """ 
    offsetmap: 2,h,w
    centermap: h,w
    """
    if device is None:
        device = torch.device('cuda')
    
    Ly0,Lx0 = masks.shape

    # get mask centers
    unique_ids = np.unique(masks)[1:]
    centers = np.zeros((len(unique_ids), 2), 'int')
    offsetmap = np.zeros((2, Ly0, Lx0))
    for i, id in enumerate(unique_ids):
        
        yi,xi = np.nonzero(masks==id)
        yi = yi.astype(np.int32)
        xi = xi.astype(np.int32)
        
        ymed = np.median(yi)
        xmed = np.median(xi)
        imin = np.argmin((xi-xmed)**2 + (yi-ymed)**2)
        
        xmed = xi[imin]
        ymed = yi[imin]
        
        offsetmap[0, yi, xi] = yi - ymed
        offsetmap[1, yi, xi] = xi - xmed
        
        centers[i,0] = xmed
        centers[i,1] = ymed

    centermap = gen_pose_target(centers, device, Ly0, Lx0, 3)
    centermap = centermap.cpu().numpy()

    comap = np.concatenate((offsetmap, centermap[np.newaxis,:,:]), axis=0)
    return comap

def masks_to_flows(masks, use_gpu=False, device=None):
    if masks.max() == 0:
        dynamics_logger.warning('empty masks!')
        return np.zeros((3, *masks.shape), 'float32')

    if use_gpu:
        if use_gpu and device is None:
            device = torch_GPU
        elif device is None:
            device = torch_CPU
    
    masks_to_flows_device = masks_to_flows_gpu
    if masks.ndim==2:
        comap = masks_to_flows_device(masks, device=device)
        return comap
    else:
        raise ValueError('masks_to_flows only takes 2D')

def labels_to_flows(labels, files=None, use_gpu=False, device=None, redo_flows=False):
    nimg = len(labels)
    if labels[0].ndim < 3:
        labels = [labels[n][np.newaxis,:,:] for n in range(nimg)]

    if labels[0].shape[0] == 1 or labels[0].ndim < 3 or redo_flows: # flows need to be recomputed
        dynamics_logger.info('computing flows for labels')
        
        # compute flows; labels are fixed here to be unique, so they need to be passed back
        # make sure labels are unique!
        labels = [fastremap.renumber(label, in_place=True)[0] for label in labels]
        #print("dy_labels:", np.array(labels).shape) #[2240,1,256,256]
        comap = [masks_to_flows(labels[n][0], use_gpu=use_gpu, device=device) for n in trange(nimg)]
        
        # concatenate labels, distance transform, vector flows, heat (boundary and mask are computed in augmentations)
        flows = [np.concatenate((labels[n], labels[n]>0.5, comap[n]), axis=0).astype(np.float32)
                    for n in range(nimg)]
        if files is not None:
            for flow, file in zip(flows, files):
                file_name = os.path.splitext(file)[0]
                tifffile.imsave(file_name+'_flows.tif', flow)
    else:
        dynamics_logger.info('flows precomputed')
        flows = [labels[n].astype(np.float32) for n in range(nimg)]
    return flows

def find_center_condidates(centermap, offsetmap, size=[256,256]):
    peak_counter = 1

    heatmap_ori = centermap
    heatmap = gaussian_filter(heatmap_ori, sigma=3)

    heatmap_left = np.zeros(heatmap.shape)
    heatmap_left[1:, :] = heatmap[:-1, :]
    heatmap_right = np.zeros(heatmap.shape)
    heatmap_right[:-1, :] = heatmap[1:, :]
    heatmap_up = np.zeros(heatmap.shape)
    heatmap_up[:, 1:] = heatmap[:, :-1]
    heatmap_down = np.zeros(heatmap.shape)
    heatmap_down[:, :-1] = heatmap[:, 1:]

    peaks_binary = np.logical_and.reduce((heatmap >= heatmap_left, heatmap >= heatmap_right, heatmap >= heatmap_up, heatmap >= heatmap_down, heatmap > 0.1))
    peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))
    peaks_with_score = [x + (heatmap_ori[x[1], x[0]], ) for x in peaks]
    id = range(peak_counter, peak_counter + len(peaks))
    peaks_with_score_and_id = [peaks_with_score[i] + (id[i], ) for i in range(len(id))]
    peak_counter = len(peaks)

    # Recover the peaks to locations in original image
    joint_candi_list = []
    for ci in range(0, peak_counter):
        joint_candi = np.zeros((1, 4))
        joint_candi[0, :] = np.array(peaks_with_score_and_id[ci])
        joint_candi_list.append(joint_candi)

    # Get the center embedding results
    embedding_list = []
    for ci in range(0, len(joint_candi_list)):
        joint_candi = joint_candi_list[ci][0, 0:2]
        embedding = np.zeros((1, 2))
        
        g_x = int(joint_candi[0])
        g_y = int(joint_candi[1])
        
        if g_x >= 0 and g_x < size[1] and g_y >= 0 and g_y < size[0]:
            offset_x = offsetmap[0, g_y, g_x]
            offset_y = offsetmap[1, g_y, g_x]
        
            embedding[0, 0] = joint_candi[0] + offset_x
            embedding[0, 1] = joint_candi[1] + offset_y
        embedding_list.append(embedding)
        
    # Convert to np array
    embedding_np_array = np.empty((0, 2))
    for ci in range(0, len(embedding_list)):
        embedding = embedding_list[ci]
        embedding_np_array = np.vstack((embedding_np_array, embedding))

    joint_candi_np_array = np.empty((0, 4))
    for ci in range(0, len(joint_candi_list)):
        joint_candi_with_type = np.zeros((1, 4))
        joint_candi = joint_candi_list[ci]
        joint_candi_with_type[0, :] = joint_candi[0, :]
        joint_candi_np_array = np.vstack((joint_candi_np_array, joint_candi_with_type))

    joint_candi_np_array_withembed = np.empty((0, 4))
    for ci in range(0, len(joint_candi_list)):
        joint_candi_with_type = np.zeros((1, 4))
        joint_candi = joint_candi_list[ci]
        joint_candi_with_type[0, 0:2] = embedding_np_array[ci]
        joint_candi_with_type[0, 2:4] = joint_candi[0, 2:]
        joint_candi_np_array_withembed = np.vstack((joint_candi_np_array_withembed, joint_candi_with_type))

    return joint_candi_np_array, joint_candi_np_array_withembed

def get_mask(center_coord, offsetmap, cp_mask):
    p_inds = np.meshgrid(np.arange(offsetmap.shape[1]), np.arange(offsetmap.shape[2]), indexing='ij')
    p = np.zeros((2, offsetmap.shape[1], offsetmap.shape[2]))
    for i in range(len(offsetmap)):
        p[i] = p_inds[i] - offsetmap[i]

    Y,X = np.nonzero(cp_mask)
    pre_center_coord = p[:, Y, X][[1,0]].transpose()
    distance_map = np.ones((pre_center_coord.shape[0], center_coord.shape[0]))*np.inf
    for i, cell_center in enumerate(center_coord[:,:2]):
        distance_map[:, i] = np.sqrt(np.sum((pre_center_coord - cell_center.reshape(-1,2))**2, axis=1))
    
    cell_index = np.argmin(distance_map, axis=1)
    mask = np.zeros((offsetmap.shape[1], offsetmap.shape[2]), dtype=np.uint16)
    mask[Y,X] = cell_index+1

    return mask

def compute_masks(offsetmap, centermap, confimap, p=None, 
                   confidence_threshold=0.0,
                   flow_threshold=0.4, interp=True, do_3D=False, 
                   min_size=300, resize=None, 
                   use_gpu=False,device=None):
    """ 
    compute masks using dynamics from offsetmap, confimap, and centermap 
    offsetmap: [2, H, W]
    centermap: [256,256]
    confimap: [256,256]
    """
    
    cp_mask = confimap > confidence_threshold 

    if np.any(cp_mask):    
        ofmap = offsetmap * cp_mask
        joint_candi_np_array, joint_candi_np_array_withembed = find_center_condidates(centermap, ofmap, size=[centermap.shape[0], centermap.shape[1]])
        if len(joint_candi_np_array_withembed) == 0:
            dynamics_logger.info('No cell pixels found.')
            shape = resize if resize is not None else confimap.shape
            mask = np.zeros(shape, np.uint16)
            return mask
        
        #calculate masks
        mask = get_mask(joint_candi_np_array_withembed, ofmap, cp_mask)
        
        if resize is not None:
            if mask.max() > 2**16-1:
                recast = True
                mask = mask.astype(np.float32)
            else:
                recast = False
                mask = mask.astype(np.uint16)
            mask = transforms.resize_image(mask, resize[0], resize[1], interpolation=cv2.INTER_NEAREST)
            if recast:
                mask = mask.astype(np.uint32)
        elif mask.max() < 2**16:
            mask = mask.astype(np.uint16)

    else: # nothing to compute, just make it compatible
        dynamics_logger.info('No cell pixels found.')
        shape = resize if resize is not None else confimap.shape
        mask = np.zeros(shape, np.uint16)
        return mask
        
    mask = utils.fill_holes_and_remove_small_masks(mask, min_size=min_size)

    if mask.dtype==np.uint32:
        dynamics_logger.warning('more than 65535 masks in image, masks returned as np.uint32')

    return mask

