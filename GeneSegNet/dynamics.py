import time, os
from scipy.ndimage.filters import maximum_filter1d
import torch
import scipy.ndimage
import numpy as np
import tifffile
from tqdm import trange
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from numba import njit, float32, int32, vectorize
import cv2
import fastremap
from scipy.ndimage.filters import gaussian_filter
import scipy.cluster.hierarchy as hcluster
from numpy.core.records import fromarrays
import math
import kornia
from morphology import Dilation2d, Erosion2d
from skimage import data, util
from skimage import measure
import csv
import itertools

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
    dilation = Dilation2d(1, 1, 3, soft_max=False).to(device)
    erosion = Erosion2d(1, 1, 3, soft_max=True).to(device)

    mask = torch.from_numpy(mask.astype(np.int32)).unsqueeze(0).unsqueeze(0).to(device)
    cell_ids = torch.unique(mask)[1:]
    
    mu = 1 
    nu = 0.003 * 255 * 255 
    num = 150
    epison = 1 
    step = 0.01

    if N < 3:
        flag = True
    else:
        flag = False

    for i in range(2):
        new_mask = mask.detach().clone()
        for cell_id in cell_ids:
            img = ((mask == cell_id)*255).float()
            LSF = ((mask == cell_id)*1).float()

            for i in range(1,num):
                if flag==True and i == 1:
                    img = erosion(img)
                    img = dilation(img)
                    img = dilation(img)
                
                LSF = level_set(LSF, img, mu, nu, epison, step)

            LSF[:][LSF[:] >= 0] = 1
            LSF[:][LSF[:] < 0] = 0

            outcoord = torch.nonzero(LSF.squeeze())
            new_mask[: , :, outcoord[:,0], outcoord[:,1]] = cell_id
            new_mask[(mask != 0) & (mask != cell_id)] = mask[(mask != 0) & (mask != cell_id)].squeeze()

        mask = new_mask
    
    mask = mask.detach().squeeze().cpu().numpy()
    return mask

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

@njit('(float64[:], int32[:], int32[:], int32, int32, int32, int32)', nogil=True)
def _extend_centers(T,y,x,ymed,xmed,Lx, niter):
    """ run diffusion from center of mask (ymed, xmed) on mask pixels (y, x)
    Parameters
    --------------
    T: float64, array
        _ x Lx array that diffusion is run in
    y: int32, array
        pixels in y inside mask
    x: int32, array
        pixels in x inside mask
    ymed: int32
        center of mask in y
    xmed: int32
        center of mask in x
    Lx: int32
        size of x-dimension of masks
    niter: int32
        number of iterations to run diffusion
    Returns
    ---------------
    T: float64, array
        amount of diffused particles at each pixel
    """

    for t in range(niter):
        T[ymed*Lx + xmed] += 1
        T[y*Lx + x] = 1/9. * (T[y*Lx + x] + T[(y-1)*Lx + x]   + T[(y+1)*Lx + x] +
                                            T[y*Lx + x-1]     + T[y*Lx + x+1] +
                                            T[(y-1)*Lx + x-1] + T[(y-1)*Lx + x+1] +
                                            T[(y+1)*Lx + x-1] + T[(y+1)*Lx + x+1])
    return T

def _extend_centers_gpu(neighbors, centers, isneighbor, Ly, Lx, n_iter=200, device=torch.device('cuda')):
    """ runs diffusion on GPU to generate flows for training images or quality control
    
    neighbors is 9 x pixels in masks, 
    centers are mask centers, 
    isneighbor is valid neighbor boolean 9 x pixels
    
    """
    if device is not None:
        device = device
    nimg = neighbors.shape[0] // 9
    pt = torch.from_numpy(neighbors).to(device)
    
    T = torch.zeros((nimg,Ly,Lx), dtype=torch.double, device=device)
    meds = torch.from_numpy(centers.astype(int)).to(device).long()
    isneigh = torch.from_numpy(isneighbor).to(device)
    for i in range(n_iter):
        T[:, meds[:,0], meds[:,1]] +=1
        Tneigh = T[:, pt[:,:,0], pt[:,:,1]]
        Tneigh *= isneigh
        T[:, pt[0,:,0], pt[0,:,1]] = Tneigh.mean(axis=1)
    
    T = torch.log(1.+ T)
    # gradient positions
    grads = T[:, pt[[2,1,4,3],:,0], pt[[2,1,4,3],:,1]]
    dy = grads[:,0] - grads[:,1]
    dx = grads[:,2] - grads[:,3]

    mu_torch = np.stack((dy.cpu().squeeze(), dx.cpu().squeeze()), axis=-2)
    return mu_torch

def masks_to_flows_gpu(masks, device=None):
    """ convert masks to flows using diffusion from center pixel
    Center of masks where diffusion starts is defined using COM
    Parameters
    -------------
    masks: int, 2D or 3D array
        labelled masks 0=NO masks; 1,2,...=mask labels
    Returns
    -------------
    mu: float, 3D or 4D array 
        flows in Y = mu[-2], flows in X = mu[-1].
        if masks are 3D, flows in Z = mu[0].
    mu_c: float, 2D or 3D array
        for each pixel, the distance to the center of the mask 
        in which it resides 
    """
    if device is None:
        device = torch.device('cuda')

    
    Ly0,Lx0 = masks.shape
    Ly, Lx = Ly0+2, Lx0+2

    masks_padded = np.zeros((Ly, Lx), np.int64)
    masks_padded[1:-1, 1:-1] = masks

    # get mask pixel neighbors
    y, x = np.nonzero(masks_padded)
    neighborsY = np.stack((y, y-1, y+1, 
                           y, y, y-1, 
                           y-1, y+1, y+1), axis=0)
    neighborsX = np.stack((x, x, x, 
                           x-1, x+1, x-1, 
                           x+1, x-1, x+1), axis=0)
    neighbors = np.stack((neighborsY, neighborsX), axis=-1)

    # get mask centers
    slices = scipy.ndimage.find_objects(masks)
    
    centers = np.zeros((masks.max(), 2), 'int')
    for i,si in enumerate(slices):
        if si is not None:
            sr,sc = si
            ly, lx = sr.stop - sr.start + 1, sc.stop - sc.start + 1
            yi,xi = np.nonzero(masks[sr, sc] == (i+1))
            yi = yi.astype(np.int32) + 1 # add padding
            xi = xi.astype(np.int32) + 1 # add padding
            ymed = np.median(yi)
            xmed = np.median(xi)
            imin = np.argmin((xi-xmed)**2 + (yi-ymed)**2)
            xmed = xi[imin]
            ymed = yi[imin]
            centers[i,0] = ymed + sr.start 
            centers[i,1] = xmed + sc.start

    # get neighbor validator (not all neighbors are in same mask)
    neighbor_masks = masks_padded[neighbors[:,:,0], neighbors[:,:,1]]
    isneighbor = neighbor_masks == neighbor_masks[0]
    ext = np.array([[sr.stop - sr.start + 1, sc.stop - sc.start + 1] for sr, sc in slices])
    n_iter = 2 * (ext.sum(axis=1)).max()
    # run diffusion
    mu = _extend_centers_gpu(neighbors, centers, isneighbor, Ly, Lx, 
                             n_iter=n_iter, device=device)

    # normalize
    mu /= (1e-20 + (mu**2).sum(axis=0)**0.5)

    # put into original image
    mu0 = np.zeros((2, Ly0, Lx0))
    mu0[:, y-1, x-1] = mu
    mu_c = np.zeros_like(mu0)
    return mu0, mu_c

def masks_to_flows_cpu(masks, device=None):
    """ convert masks to flows using diffusion from center pixel
    Center of masks where diffusion starts is defined to be the 
    closest pixel to the median of all pixels that is inside the 
    mask. Result of diffusion is converted into flows by computing
    the gradients of the diffusion density map. 
    Parameters
    -------------
    masks: int, 2D array
        labelled masks 0=NO masks; 1,2,...=mask labels
    Returns
    -------------
    mu: float, 3D array 
        flows in Y = mu[-2], flows in X = mu[-1].
        if masks are 3D, flows in Z = mu[0].
    mu_c: float, 2D array
        for each pixel, the distance to the center of the mask 
        in which it resides 
    """
    
    Ly, Lx = masks.shape
    mu = np.zeros((2, Ly, Lx), np.float64)
    mu_c = np.zeros((Ly, Lx), np.float64)
    
    nmask = masks.max()
    slices = scipy.ndimage.find_objects(masks)
    dia = utils.diameters(masks)[0]
    s2 = (.15 * dia)**2
    for i,si in enumerate(slices):
        if si is not None:
            sr,sc = si
            ly, lx = sr.stop - sr.start + 1, sc.stop - sc.start + 1
            y,x = np.nonzero(masks[sr, sc] == (i+1))
            y = y.astype(np.int32) + 1
            x = x.astype(np.int32) + 1
            ymed = np.median(y)
            xmed = np.median(x)
            imin = np.argmin((x-xmed)**2 + (y-ymed)**2)
            xmed = x[imin]
            ymed = y[imin]
            
            d2 = (x-xmed)**2 + (y-ymed)**2
            mu_c[sr.start+y-1, sc.start+x-1] = np.exp(-d2/s2)

            niter = 2*np.int32(np.ptp(x) + np.ptp(y))
            T = np.zeros((ly+2)*(lx+2), np.float64)
            T = _extend_centers(T, y, x, ymed, xmed, np.int32(lx), np.int32(niter))
            T[(y+1)*lx + x+1] = np.log(1.+T[(y+1)*lx + x+1])

            dy = T[(y+1)*lx + x] - T[(y-1)*lx + x]
            dx = T[y*lx + x+1] - T[y*lx + x-1]
            mu[:, sr.start+y-1, sc.start+x-1] = np.stack((dy,dx))

    mu /= (1e-20 + (mu**2).sum(axis=0)**0.5)

    return mu, mu_c

def masks_to_flows(masks, use_gpu=False, device=None):
    """ convert masks to flows using diffusion from center pixel

    Center of masks where diffusion starts is defined to be the 
    closest pixel to the median of all pixels that is inside the 
    mask. Result of diffusion is converted into flows by computing
    the gradients of the diffusion density map. 

    Parameters
    -------------

    masks: int, 2D or 3D array
        labelled masks 0=NO masks; 1,2,...=mask labels

    Returns
    -------------

    mu: float, 3D or 4D array 
        flows in Y = mu[-2], flows in X = mu[-1].
        if masks are 3D, flows in Z = mu[0].

    mu_c: float, 2D or 3D array
        for each pixel, the distance to the center of the mask 
        in which it resides 

    """
    if masks.max() == 0:
        dynamics_logger.warning('empty masks!')
        return np.zeros((2, *masks.shape), 'float32')

    if use_gpu:
        if use_gpu and device is None:
            device = torch_GPU
        elif device is None:
            device = torch_CPU
        masks_to_flows_device = masks_to_flows_gpu
    else:
        masks_to_flows_device = masks_to_flows_cpu
        
    if masks.ndim==3:
        Lz, Ly, Lx = masks.shape
        mu = np.zeros((3, Lz, Ly, Lx), np.float32)
        for z in range(Lz):
            mu0 = masks_to_flows_device(masks[z], device=device)[0]
            mu[[1,2], z] += mu0
        for y in range(Ly):
            mu0 = masks_to_flows_device(masks[:,y], device=device)[0]
            mu[[0,2], :, y] += mu0
        for x in range(Lx):
            mu0 = masks_to_flows_device(masks[:,:,x], device=device)[0]
            mu[[0,1], :, :, x] += mu0
        return mu
    elif masks.ndim==2:
        mu, mu_c = masks_to_flows_device(masks, device=device)
        return mu

    else:
        raise ValueError('masks_to_flows only takes 2D or 3D arrays')

# def labels_to_flows(labels, files=None, use_gpu=False, device=None, redo_flows=False):
#     nimg = len(labels)
#     if labels[0].ndim < 3:
#         labels = [labels[n][np.newaxis,:,:] for n in range(nimg)]

#     if labels[0].shape[0] == 1 or labels[0].ndim < 3 or redo_flows: # flows need to be recomputed
#         dynamics_logger.info('computing flows for labels')
        
#         # compute flows; labels are fixed here to be unique, so they need to be passed back
#         # make sure labels are unique!
#         labels = [fastremap.renumber(label, in_place=True)[0] for label in labels]
#         #print("dy_labels:", np.array(labels).shape) #[2240,1,256,256]
#         comap = [masks_to_flows(labels[n][0], use_gpu=use_gpu, device=device) for n in trange(nimg)]
        
#         # concatenate labels, distance transform, vector flows, heat (boundary and mask are computed in augmentations)
#         flows = [np.concatenate((labels[n], labels[n]>0.5, comap[n]), axis=0).astype(np.float32)
#                     for n in range(nimg)]
#         if files is not None:
#             for flow, file in zip(flows, files):
#                 file_name = os.path.splitext(file)[0]
#                 tifffile.imsave(file_name+'_flows.tif', flow)
#     else:
#         dynamics_logger.info('flows precomputed')
#         flows = [labels[n].astype(np.float32) for n in range(nimg)]
#     return flows

def labels_to_flows(labels, files=None, use_gpu=False, device=None, redo_flows=False):
    """ convert labels (list of masks or flows) to flows for training model 

    if files is not None, flows are saved to files to be reused

    Parameters
    --------------

    labels: list of ND-arrays
        labels[k] can be 2D or 3D, if [3 x Ly x Lx] then it is assumed that flows were precomputed.
        Otherwise labels[k][0] or labels[k] (if 2D) is used to create flows and cell probabilities.

    Returns
    --------------

    flows: list of [4 x Ly x Lx] arrays
        flows[k][0] is labels[k], flows[k][1] is cell distance transform, flows[k][2] is Y flow,
        flows[k][3] is X flow, and flows[k][4] is heat distribution

    """
    nimg = len(labels)
    if labels[0].ndim < 3:
        labels = [labels[n][np.newaxis,:,:] for n in range(nimg)]

    if labels[0].shape[0] == 1 or labels[0].ndim < 3 or redo_flows: # flows need to be recomputed
        
        dynamics_logger.info('computing flows for labels')
        
        # compute flows; labels are fixed here to be unique, so they need to be passed back
        # make sure labels are unique!
        labels = [fastremap.renumber(label, in_place=True)[0] for label in labels]
        veci = [masks_to_flows(labels[n][0],use_gpu=use_gpu, device=device) for n in trange(nimg)]
        
        # concatenate labels, distance transform, vector flows, heat (boundary and mask are computed in augmentations)
        flows = [np.concatenate((labels[n], labels[n]>0.5, veci[n]), axis=0).astype(np.float32)
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
    heatmap = gaussian_filter(heatmap_ori, sigma=2)

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


def get_masks_from_offset(p, iscell=None, rpad=20):
    """ create masks using pixel convergence after running dynamics
    
    Makes a histogram of final pixel locations p, initializes masks 
    at peaks of histogram and extends the masks from the peaks so that
    they include all pixels with more than 2 final pixels p. Discards 
    masks with flow errors greater than the threshold. 
    Parameters
    ----------------
    p: float32, 3D or 4D array
        final locations of each pixel after dynamics,
        size [axis x Ly x Lx] or [axis x Lz x Ly x Lx].
    iscell: bool, 2D or 3D array
        if iscell is not None, set pixels that are 
        iscell False to stay in their original location.
    rpad: int (optional, default 20)
        histogram edge padding
    threshold: float (optional, default 0.4)
        masks with flow error greater than threshold are discarded 
        (if flows is not None)
    flows: float, 3D or 4D array (optional, default None)
        flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]. If flows
        is not None, then masks with inconsistent flows are removed using 
        `remove_bad_flow_masks`.
    Returns
    ---------------
    M0: int, 2D or 3D array
        masks with inconsistent flow masks removed, 
        0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]
    
    """
    
    pflows = []
    edges = []
    shape0 = p.shape[1:]
    dims = len(p)
    if iscell is not None:
        if dims==3:
            inds = np.meshgrid(np.arange(shape0[0]), np.arange(shape0[1]),
                np.arange(shape0[2]), indexing='ij')
        elif dims==2:
            inds = np.meshgrid(np.arange(shape0[0]), np.arange(shape0[1]),
                     indexing='ij')
        for i in range(dims):
            p[i, ~iscell] = inds[i][~iscell]

    for i in range(dims):
        pflows.append(p[i].flatten().astype('int32'))
        edges.append(np.arange(-.5-rpad, shape0[i]+.5+rpad, 1))

    h,_ = np.histogramdd(tuple(pflows), bins=edges)
    hmax = h.copy()
    for i in range(dims):
        hmax = maximum_filter1d(hmax, 5, axis=i)

    seeds = np.nonzero(np.logical_and(h-hmax>-1e-6, h>10))
    Nmax = h[seeds]
    isort = np.argsort(Nmax)[::-1]
    for s in seeds:
        s = s[isort]

    pix = list(np.array(seeds).T)

    shape = h.shape
    if dims==3:
        expand = np.nonzero(np.ones((3,3,3)))
    else:
        expand = np.nonzero(np.ones((3,3)))
    for e in expand:
        e = np.expand_dims(e,1)

    for iter in range(5):
        for k in range(len(pix)):
            if iter==0:
                pix[k] = list(pix[k])
            newpix = []
            iin = []
            for i,e in enumerate(expand):
                epix = e[:,np.newaxis] + np.expand_dims(pix[k][i], 0) - 1
                epix = epix.flatten()
                iin.append(np.logical_and(epix>=0, epix<shape[i]))
                newpix.append(epix)
            iin = np.all(tuple(iin), axis=0)
            for p in newpix:
                p = p[iin]
            newpix = tuple(newpix)
            igood = h[newpix]>2
            for i in range(dims):
                pix[k][i] = newpix[i][igood]
            if iter==4:
                pix[k] = tuple(pix[k])
    
    M = np.zeros(h.shape, np.uint32)
    for k in range(len(pix)):
        M[pix[k]] = 1+k
        
    for i in range(dims):
        pflows[i] = pflows[i] + rpad
    M0 = M[tuple(pflows)]

    # remove big masks
    uniq, counts = fastremap.unique(M0, return_counts=True)
    big = np.prod(shape0)
    bigc = uniq[counts > big]
    if len(bigc) > 0 and (len(bigc)>1 or bigc[0]!=0):
        M0 = fastremap.mask(M0, bigc)
    fastremap.renumber(M0, in_place=True) #convenient to guarantee non-skipped labels
    M0 = np.reshape(M0, shape0)
    return M0

@njit(['(int16[:,:,:], float32[:], float32[:], float32[:,:])', 
        '(float32[:,:,:], float32[:], float32[:], float32[:,:])'], cache=True)
def map_coordinates(I, yc, xc, Y):
    """
    bilinear interpolation of image 'I' in-place with ycoordinates yc and xcoordinates xc to Y
    
    Parameters
    -------------
    I : C x Ly x Lx
    yc : ni
        new y coordinates
    xc : ni
        new x coordinates
    Y : C x ni
        I sampled at (yc,xc)
    """
    C,Ly,Lx = I.shape
    yc_floor = yc.astype(np.int32)
    xc_floor = xc.astype(np.int32)
    yc = yc - yc_floor
    xc = xc - xc_floor
    for i in range(yc_floor.shape[0]):
        yf = min(Ly-1, max(0, yc_floor[i]))
        xf = min(Lx-1, max(0, xc_floor[i]))
        yf1= min(Ly-1, yf+1)
        xf1= min(Lx-1, xf+1)
        y = yc[i]
        x = xc[i]
        for c in range(C):
            Y[c,i] = (np.float32(I[c, yf, xf]) * (1 - y) * (1 - x) +
                      np.float32(I[c, yf, xf1]) * (1 - y) * x +
                      np.float32(I[c, yf1, xf]) * y * (1 - x) +
                      np.float32(I[c, yf1, xf1]) * y * x )


def steps2D_interp(p, dP, niter, use_gpu=False, device=None):
    shape = dP.shape[1:]
    if use_gpu:
        if device is None:
            device = torch_GPU
        shape = np.array(shape)[[1,0]].astype('float')-1  # Y and X dimensions (dP is 2.Ly.Lx), flipped X-1, Y-1
        pt = torch.from_numpy(p[[1,0]].T).float().to(device).unsqueeze(0).unsqueeze(0) # p is n_points by 2, so pt is [1 1 2 n_points]
        im = torch.from_numpy(dP[[1,0]]).float().to(device).unsqueeze(0) #covert flow numpy array to tensor on GPU, add dimension 
        # normalize pt between  0 and  1, normalize the flow
        for k in range(2): 
            im[:,k,:,:] *= 2./shape[k]
            pt[:,:,:,k] /= shape[k]
            
        # normalize to between -1 and 1
        pt = pt*2-1 
        
        #here is where the stepping happens
        for t in range(niter):
            # align_corners default is False, just added to suppress warning
            dPt = torch.nn.functional.grid_sample(im, pt, align_corners=False)
            
            for k in range(2): #clamp the final pixel locations
                pt[:,:,:,k] = torch.clamp(pt[:,:,:,k] + dPt[:,k,:,:], -1., 1.)
            

        #undo the normalization from before, reverse order of operations 
        pt = (pt+1)*0.5
        for k in range(2): 
            pt[:,:,:,k] *= shape[k]        
        
        p =  pt[:,:,:,[1,0]].cpu().numpy().squeeze().T
        return p

    else:
        dPt = np.zeros(p.shape, np.float32)
            
        for t in range(niter):
            map_coordinates(dP.astype(np.float32), p[0], p[1], dPt)
            for k in range(len(p)):
                p[k] = np.minimum(shape[k]-1, np.maximum(0, p[k] + dPt[k]))
        return p


@njit('(float32[:,:,:,:],float32[:,:,:,:], int32[:,:], int32)', nogil=True)
def steps3D(p, dP, inds, niter):
    """ run dynamics of pixels to recover masks in 3D
    
    Euler integration of dynamics dP for niter steps

    Parameters
    ----------------

    p: float32, 4D array
        pixel locations [axis x Lz x Ly x Lx] (start at initial meshgrid)

    dP: float32, 4D array
        flows [axis x Lz x Ly x Lx]

    inds: int32, 2D array
        non-zero pixels to run dynamics on [npixels x 3]

    niter: int32
        number of iterations of dynamics to run

    Returns
    ---------------

    p: float32, 4D array
        final locations of each pixel after dynamics

    """
    shape = p.shape[1:]
    for t in range(niter):
        #pi = p.astype(np.int32)
        for j in range(inds.shape[0]):
            z = inds[j,0]
            y = inds[j,1]
            x = inds[j,2]
            p0, p1, p2 = int(p[0,z,y,x]), int(p[1,z,y,x]), int(p[2,z,y,x])
            p[0,z,y,x] = min(shape[0]-1, max(0, p[0,z,y,x] + dP[0,p0,p1,p2]))
            p[1,z,y,x] = min(shape[1]-1, max(0, p[1,z,y,x] + dP[1,p0,p1,p2]))
            p[2,z,y,x] = min(shape[2]-1, max(0, p[2,z,y,x] + dP[2,p0,p1,p2]))
    return p

@njit('(float32[:,:,:], float32[:,:,:], int32[:,:], int32)', nogil=True)
def steps2D(p, dP, inds, niter):
    """ run dynamics of pixels to recover masks in 2D
    
    Euler integration of dynamics dP for niter steps

    Parameters
    ----------------

    p: float32, 3D array
        pixel locations [axis x Ly x Lx] (start at initial meshgrid)

    dP: float32, 3D array
        flows [axis x Ly x Lx]

    inds: int32, 2D array
        non-zero pixels to run dynamics on [npixels x 2]

    niter: int32
        number of iterations of dynamics to run

    Returns
    ---------------

    p: float32, 3D array
        final locations of each pixel after dynamics

    """
    shape = p.shape[1:]
    for t in range(niter):
        for j in range(inds.shape[0]):
            # starting coordinates
            y = inds[j,0]
            x = inds[j,1]
            p0, p1 = int(p[0,y,x]), int(p[1,y,x])
            step = dP[:,p0,p1]
            for k in range(p.shape[0]):
                p[k,y,x] = min(shape[k]-1, max(0, p[k,y,x] + step[k]))
    return p

def follow_flows(dP, mask=None, niter=200, interp=True, use_gpu=True, device=None):
    """ define pixels and run dynamics to recover masks in 2D
    
    Pixels are meshgrid. Only pixels with non-zero cell-probability
    are used (as defined by inds)

    Parameters
    ----------------

    dP: float32, 3D or 4D array
        flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]
    
    mask: (optional, default None)
        pixel mask to seed masks. Useful when flows have low magnitudes.

    niter: int (optional, default 200)
        number of iterations of dynamics to run

    interp: bool (optional, default True)
        interpolate during 2D dynamics (not available in 3D) 
        (in previous versions + paper it was False)

    use_gpu: bool (optional, default False)
        use GPU to run interpolated dynamics (faster than CPU)


    Returns
    ---------------

    p: float32, 3D or 4D array
        final locations of each pixel after dynamics; [axis x Ly x Lx] or [axis x Lz x Ly x Lx]

    inds: int32, 3D or 4D array
        indices of pixels used for dynamics; [axis x Ly x Lx] or [axis x Lz x Ly x Lx]

    """
    shape = np.array(dP.shape[1:]).astype(np.int32)
    niter = np.uint32(niter)
    if len(shape)>2:
        p = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]),
                np.arange(shape[2]), indexing='ij')
        p = np.array(p).astype(np.float32)
        # run dynamics on subset of pixels
        inds = np.array(np.nonzero(np.abs(dP[0])>1e-3)).astype(np.int32).T
        p = steps3D(p, dP, inds, niter)
    else:
        p = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        p = np.array(p).astype(np.float32)

        inds = np.array(np.nonzero(np.abs(dP[0])>1e-3)).astype(np.int32).T
        
        if inds.ndim < 2 or inds.shape[0] < 5:
            dynamics_logger.warning('WARNING: no mask pixels found')
            return p, None
        
        if not interp:
            p = steps2D(p, dP.astype(np.float32), inds, niter)
            
        else:
            p_interp = steps2D_interp(p[:,inds[:,0], inds[:,1]], dP, niter, use_gpu=use_gpu, device=device)            
            p[:,inds[:,0],inds[:,1]] = p_interp
    return p, inds

def remove_bad_flow_masks(masks, flows, threshold=400, use_gpu=False, device=None):
    """ remove masks which have inconsistent flows 
    
    Uses metrics.flow_error to compute flows from predicted masks 
    and compare flows to predicted flows from network. Discards 
    masks with flow errors greater than the threshold.

    Parameters
    ----------------

    masks: int, 2D or 3D array
        labelled masks, 0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]

    flows: float, 3D or 4D array
        flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]

    threshold: float (optional, default 0.4)
        masks with flow error greater than threshold are discarded.

    Returns
    ---------------

    masks: int, 2D or 3D array
        masks with inconsistent flow masks removed, 
        0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]
    
    """
    merrors, _ = metrics.flow_error(masks, flows, use_gpu, device)
    # print("merrors:", merrors)
    badi = 1+(merrors>threshold).nonzero()[0]
    masks[np.isin(masks, badi)] = 0
    return masks

def get_masks(p, iscell=None, rpad=20):
    """ create masks using pixel convergence after running dynamics
    
    Makes a histogram of final pixel locations p, initializes masks 
    at peaks of histogram and extends the masks from the peaks so that
    they include all pixels with more than 2 final pixels p. Discards 
    masks with flow errors greater than the threshold. 
    Parameters
    ----------------
    p: float32, 3D or 4D array
        final locations of each pixel after dynamics,
        size [axis x Ly x Lx] or [axis x Lz x Ly x Lx].
    iscell: bool, 2D or 3D array
        if iscell is not None, set pixels that are 
        iscell False to stay in their original location.
    rpad: int (optional, default 20)
        histogram edge padding
    threshold: float (optional, default 0.4)
        masks with flow error greater than threshold are discarded 
        (if flows is not None)
    flows: float, 3D or 4D array (optional, default None)
        flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]. If flows
        is not None, then masks with inconsistent flows are removed using 
        `remove_bad_flow_masks`.
    Returns
    ---------------
    M0: int, 2D or 3D array
        masks with inconsistent flow masks removed, 
        0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]
    
    """
    
    pflows = []
    edges = []
    shape0 = p.shape[1:]
    dims = len(p)
    if iscell is not None:
        if dims==3:
            inds = np.meshgrid(np.arange(shape0[0]), np.arange(shape0[1]),
                np.arange(shape0[2]), indexing='ij')
        elif dims==2:
            inds = np.meshgrid(np.arange(shape0[0]), np.arange(shape0[1]),
                     indexing='ij')
        for i in range(dims):
            p[i, ~iscell] = inds[i][~iscell]

    for i in range(dims):
        pflows.append(p[i].flatten().astype('int32'))
        edges.append(np.arange(-.5-rpad, shape0[i]+.5+rpad, 1))

    h,_ = np.histogramdd(tuple(pflows), bins=edges)
    hmax = h.copy()
    for i in range(dims):
        hmax = maximum_filter1d(hmax, 5, axis=i)

    seeds = np.nonzero(np.logical_and(h-hmax>-1e-6, h>10))
    Nmax = h[seeds]
    isort = np.argsort(Nmax)[::-1]
    for s in seeds:
        s = s[isort]

    pix = list(np.array(seeds).T)

    shape = h.shape
    if dims==3:
        expand = np.nonzero(np.ones((3,3,3)))
    else:
        expand = np.nonzero(np.ones((3,3)))
    for e in expand:
        e = np.expand_dims(e,1)

    for iter in range(5):
        for k in range(len(pix)):
            if iter==0:
                pix[k] = list(pix[k])
            newpix = []
            iin = []
            for i,e in enumerate(expand):
                epix = e[:,np.newaxis] + np.expand_dims(pix[k][i], 0) - 1
                epix = epix.flatten()
                iin.append(np.logical_and(epix>=0, epix<shape[i]))
                newpix.append(epix)
            iin = np.all(tuple(iin), axis=0)
            for p in newpix:
                p = p[iin]
            newpix = tuple(newpix)
            igood = h[newpix]>2
            for i in range(dims):
                pix[k][i] = newpix[i][igood]
            if iter==4:
                pix[k] = tuple(pix[k])
    
    M = np.zeros(h.shape, np.uint32)
    for k in range(len(pix)):
        M[pix[k]] = 1+k
        
    for i in range(dims):
        pflows[i] = pflows[i] + rpad
    M0 = M[tuple(pflows)]

    # remove big masks
    uniq, counts = fastremap.unique(M0, return_counts=True)
    big = np.prod(shape0) * 0.4
    bigc = uniq[counts > big]
    if len(bigc) > 0 and (len(bigc)>1 or bigc[0]!=0):
        M0 = fastremap.mask(M0, bigc)
    fastremap.renumber(M0, in_place=True) #convenient to guarantee non-skipped labels
    M0 = np.reshape(M0, shape0)
    return M0

def compute_masks(offsetmap, confimap, p=None, niter=200, 
                   confidence_threshold=0.9,
                   flow_threshold=0.4, interp=True, do_3D=False, 
                   min_size=15, resize=None, 
                   use_gpu=False,device=None):
    """ compute masks using dynamics from offsetmap, confimap """
    
    cp_mask = confimap > confidence_threshold 

    if np.any(cp_mask): #mask at this point is a cell cluster binary map, not labels     
        # follow flows
        if p is None:
            p, inds = follow_flows(offsetmap * cp_mask / 5., niter=niter, interp=interp, 
                                            use_gpu=use_gpu, device=device)
            if inds is None:
                dynamics_logger.info('No cell pixels found.')
                shape = resize if resize is not None else confimap.shape
                mask = np.zeros(shape, np.uint16)
                p = np.zeros((len(shape), *shape), np.uint16)
                return mask, p
        
        #calculate masks
        mask = get_masks(p, iscell=cp_mask)
            
        # flow thresholding factored out of get_masks
        if not do_3D:
            shape0 = p.shape[1:]
            if mask.max()>0 and flow_threshold is not None and flow_threshold > 0:
                # make sure labels are unique at output of get_masks
                mask = remove_bad_flow_masks(mask, offsetmap, threshold=flow_threshold, use_gpu=use_gpu, device=device)
        
        if resize is not None:
            #if verbose:
            #    dynamics_logger.info(f'resizing output with resize = {resize}')
            if mask.max() > 2**16-1:
                recast = True
                mask = mask.astype(np.float32)
            else:
                recast = False
                mask = mask.astype(np.uint16)
            mask = transforms.resize_image(mask, resize[0], resize[1], interpolation=cv2.INTER_NEAREST)
            if recast:
                mask = mask.astype(np.uint32)
            Ly,Lx = mask.shape
        elif mask.max() < 2**16:
            mask = mask.astype(np.uint16)

    else: # nothing to compute, just make it compatible
        dynamics_logger.info('No cell pixels found.')
        shape = resize if resize is not None else confimap.shape
        mask = np.zeros(shape, np.uint16)
        p = np.zeros((len(shape), *shape), np.uint16)
        return mask, p
 
    mask = utils.fill_holes_and_remove_small_masks(mask, min_size=min_size)

    if mask.dtype==np.uint32:
        dynamics_logger.warning('more than 65535 masks in image, masks returned as np.uint32')

    return mask, p

def calculate_iou(m1, m2):
    # Initialize a 2D array to store the IoU values
    iou_values = np.zeros((m2.max(), m1.max()))
    # print("iou_values shape:", iou_values.shape)
    # Calculate the IoU for each pair of masks
    for i in np.unique(m2)[1:]:
        for j in np.unique(m1)[1:]:
            # Calculate intersection and union
            intersection = np.logical_and(m2 == i, m1 == j).sum()
            union = np.logical_or(m2 == i, m1 == j).sum()

            # Calculate IoU and store it in the array
            if union == 0:
                iou_values[i-1, j-1] = 0.0
            else:
                iou_values[i-1, j-1] = intersection / union

    return iou_values

def compute_perpendicular_line(coord1, coord2):
    # Calculate the midpoint
    mid_x = (coord1[0] + coord2[0]) / 2
    mid_y = (coord1[1] + coord2[1]) / 2

    # Calculate the slope of the original line
    if coord1[0] == coord2[0]:  # The line is vertical
        # The perpendicular line is horizontal, so its slope is 0
        # and its y-intercept is the y-coordinate of the midpoint
        slope = 0
        y_intercept = mid_y
    else:
        original_slope = (coord2[1] - coord1[1]) / (coord2[0] - coord1[0])
        
        # Calculate the slope of the perpendicular line
        if original_slope == 0:  # The original line is horizontal
            # The perpendicular line is vertical, so its slope is undefined
            # In this case, we return the x-coordinate of the midpoint
            return mid_x
        else:
            slope = -1 / original_slope
        
        # Calculate the y-intercept of the perpendicular line
        # using the formula: b = y - mx
        y_intercept = mid_y - slope * mid_x

    return slope, y_intercept

def refine_mask(m1, m2):
    iou_matrix = calculate_iou(m1,m2)
    refine_mask = np.zeros((m1.shape[0], m1.shape[1])).astype(np.uint16)
    
    #Check if column is all 0
    cellid = np.maximum(m1.max(), m2.max()) + 1
    zero_col_index = np.where(~iou_matrix.any(axis=0))[0]
    if len(zero_col_index) != 0:
        for inds in zero_col_index:
            refine_mask[m1 == (inds+1)] = cellid
            cellid += 1

    #Check if two cells are split into one
    inds = np.argmax(iou_matrix, axis=1)
    visited = set()
    dups = {x for x in inds if x in visited or (visited.add(x) or False)}
    print(dups)  # {1, 5}
    for dup in list(dups):
        indexs = np.where(inds==dup)[0]
        centroids = []
        for index in indexs:
            mask = (m2 == (index+1))
            y, x = np.nonzero(mask)
            maxy, miny = y.max(), y.min()
            maxx, minx = x.max(), x.min()
            centroid_y, centroid_x = (maxy + miny)//2, (maxx + minx)//2
            centroids.append([centroid_x, centroid_y])

        centroids_comb = list(itertools.combinations(centroids, 2))

    for inds_m2, iou_line in enumerate(iou_matrix):
        inds_m1 = np.argmax(iou_line)
        # print("inds:", inds)
        if iou_line[inds_m1] > 0.1:
            m1_area = (m1 == (inds_m1+1)).sum()
            m2_area = (m2 == (inds_m2+1)).sum()
            if m2_area < m1_area:
                refine_mask[m1 == (inds_m1+1)] = (inds_m2+1)
            else:
                refine_mask[m2 == (inds_m2+1)] = (inds_m2+1)
        if iou_line[inds_m1] == 0.0:
            refine_mask[m2 == (inds_m2+1)] = (inds_m2+1)
        
    refine_mask = fastremap.renumber(refine_mask, in_place=True)[0]

    return refine_mask

def fill_holes(mask):
    h, w = mask.shape
    mask_new = np.zeros((h,w), np.uint8)
    for unid in np.unique(mask)[1:]:
        bmask = (mask == unid).astype(np.uint8)
        bmask_cp = bmask.copy()
        zmask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(bmask_cp, zmask, (0,0), 255)
        bmask_cp_inv = cv2.bitwise_not(bmask_cp)
        out = bmask | bmask_cp_inv
        if out.sum() < 0.9 * h * w:
            mask_new[out.astype(np.bool)] = unid
    return mask_new


def get_mask(center_coord, p, cp_mask, flow_threshold=None):
    Y,X = np.nonzero(cp_mask)
    pre_center_coord = p[:, Y, X][[1,0]].transpose()
    distance_map = np.ones((pre_center_coord.shape[0], center_coord.shape[0]))*np.inf
    for i, cell_center in enumerate(center_coord[:,:2]):
        distance_map[:, i] = np.sqrt(np.sum((pre_center_coord - cell_center.reshape(-1,2))**2, axis=1))
    
    cell_index = np.argmin(distance_map, axis=1)
    mask = np.zeros((cp_mask.shape[0], cp_mask.shape[1]), dtype=np.uint16)
    mask[Y,X] = cell_index+1

    #Compute the connected components
    mask_new = measure.label(mask, connectivity = mask.ndim) #[256,256]
    regions = measure.regionprops(mask_new)
    mask = mask_new.copy()
    for region in regions:
        if region.bbox_area > 0.9 * mask.shape[0] * mask.shape[1]:
            mask[mask_new == (region.label)] = 0
        
        if region.area < 50 or region.bbox[2] - region.bbox[0] < 5 or region.bbox[3] - region.bbox[1] < 5:
            mask[mask_new == (region.label)] = 0
    
    mask = fastremap.renumber(mask, in_place=True)[0]
    
    return mask

def compute_masks_from_offset(offsetmap, centermap, confimap, p=None, 
                   confidence_threshold=0.0,
                   flow_threshold=400, interp=True, do_3D=False, 
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

        p_inds = np.meshgrid(np.arange(offsetmap.shape[1]), np.arange(offsetmap.shape[2]), indexing='ij')
        p = np.zeros((2, offsetmap.shape[1], offsetmap.shape[2]))
        for i in range(len(offsetmap)):
            p[i] = p_inds[i] - offsetmap[i]
        
        shape = resize if resize is not None else confimap.shape
        mask_center = np.zeros(shape, np.uint16)
        if len(joint_candi_np_array_withembed) != 0:
            mask_center = get_mask(joint_candi_np_array_withembed, p, cp_mask, flow_threshold)
            # print("cellcenter", np.unique(mask_center))
        
        mask_offset = get_masks_from_offset(p, cp_mask)
    
        if mask_offset.max()>0 and flow_threshold is not None and flow_threshold > 0:
            # make sure labels are unique at output of get_masks
            mask_offset = remove_bad_flow_masks(mask_offset, offsetmap, threshold=flow_threshold, use_gpu=True, device="cuda:0")
        
        mask_offset = utils.fill_holes_and_remove_small_masks(mask_offset, min_size=min_size)
        # print("mask_offset:", np.unique(mask_offset))
        
        if mask_center.max() != 0 and mask_offset.max() != 0:
            mask = refine_mask(mask_center, mask_offset)
        elif mask_offset.max() != 0:
            mask = mask_offset
        elif mask_center.max() != 0:
            mask = mask_center
        else:
            mask = np.zeros(shape, np.uint16)
        
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

