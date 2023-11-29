import numpy as np
import utils, dynamics, Gseg_io
from numba import jit
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import convolve, mean
from natsort import natsorted
import glob
import os
import cv2

def mask_ious(masks_true, masks_pred):
    """ return best-matched masks """
    iou = _intersection_over_union(masks_true, masks_pred)[1:,1:]
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= 0.5).astype(float) - iou / (2*n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    iout = np.zeros(masks_true.max())
    iout[true_ind] = iou[true_ind,pred_ind]
    preds = np.zeros(masks_true.max(), 'int')
    preds[true_ind] = pred_ind+1
    return iout, preds

def boundary_scores(masks_true, masks_pred, scales):
    """ boundary precision / recall / Fscore """
    diams = [utils.diameters(lbl)[0] for lbl in masks_true]
    precision = np.zeros((len(scales), len(masks_true)))
    recall = np.zeros((len(scales), len(masks_true)))
    fscore = np.zeros((len(scales), len(masks_true)))
    for j, scale in enumerate(scales):
        #print("metrics_scale:", scale)
        for n in range(len(masks_true)):
            #print("metrics_diam_beforescale:", diams[n])
            diam = max(1, scale * diams[n])
            #print("metrics_diam_afterscale:", diam)
            rs, ys, xs = utils.circleMask([int(np.ceil(diam)), int(np.ceil(diam))])
            filt = (rs <= diam).astype(np.float32)
            otrue = utils.masks_to_outlines(masks_true[n])
            otrue = convolve(otrue, filt)
            opred = utils.masks_to_outlines(masks_pred[n])
            opred = convolve(opred, filt)
            tp = np.logical_and(otrue==1, opred==1).sum()
            fp = np.logical_and(otrue==0, opred==1).sum()
            fn = np.logical_and(otrue==1, opred==0).sum()
            precision[j,n] = tp / (tp + fp)
            recall[j,n] = tp / (tp + fn)
        fscore[j] = 2 * precision[j] * recall[j] / (precision[j] + recall[j])
    return precision, recall, fscore


def aggregated_jaccard_index(masks_true, masks_pred):
    """ AJI = intersection of all matched masks / union of all masks 
    
    Parameters
    ------------
    
    masks_true: list of ND-arrays (int) or ND-array (int) 
        where 0=NO masks; 1,2... are mask labels
    masks_pred: list of ND-arrays (int) or ND-array (int) 
        ND-array (int) where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    aji : aggregated jaccard index for each set of masks

    """

    aji = np.zeros(len(masks_true))
    for n in range(len(masks_true)):
        iout, preds = mask_ious(masks_true[n], masks_pred[n])
        inds = np.arange(0, masks_true[n].max(), 1, int)
        overlap = _label_overlap(masks_true[n], masks_pred[n])
        union = np.logical_or(masks_true[n]>0, masks_pred[n]>0).sum()
        overlap = overlap[inds[preds>0]+1, preds[preds>0].astype(int)]
        aji[n] = overlap.sum() / union
    return aji 


def average_precision(masks_true, masks_pred, threshold=[0.25, 0.75, 0.9]):
    """ average precision estimation: AP = TP / (TP + FP + FN)

    This function is based heavily on the *fast* stardist matching functions
    (https://github.com/mpicbg-csbd/stardist/blob/master/stardist/matching.py)

    Parameters
    ------------
    
    masks_true: list of ND-arrays (int) or ND-array (int) 
        where 0=NO masks; 1,2... are mask labels
    masks_pred: list of ND-arrays (int) or ND-array (int) 
        ND-array (int) where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    ap: array [len(masks_true) x len(threshold)]
        average precision at thresholds
    tp: array [len(masks_true) x len(threshold)]
        number of true positives at thresholds
    fp: array [len(masks_true) x len(threshold)]
        number of false positives at thresholds
    fn: array [len(masks_true) x len(threshold)]
        number of false negatives at thresholds

    """
    not_list = False
    if not isinstance(masks_true, list):
        masks_true = [masks_true]
        masks_pred = [masks_pred]
        not_list = True
    if not isinstance(threshold, list) and not isinstance(threshold, np.ndarray):
        threshold = [threshold]
    
    if len(masks_true) != len(masks_pred):
        raise ValueError('metrics.average_precision requires len(masks_true)==len(masks_pred)')

    ap  = np.zeros((len(masks_true), len(threshold)), np.float32)
    tp  = np.zeros((len(masks_true), len(threshold)), np.float32)
    fp  = np.zeros((len(masks_true), len(threshold)), np.float32)
    fn  = np.zeros((len(masks_true), len(threshold)), np.float32)
    n_true = np.array(list(map(np.max, masks_true)))
    n_pred = np.array(list(map(np.max, masks_pred)))
    
    for n in range(len(masks_true)):
        #_,mt = np.reshape(np.unique(masks_true[n], return_index=True), masks_pred[n].shape)
        if n_pred[n] > 0:
            iou = _intersection_over_union(masks_true[n], masks_pred[n])[1:, 1:]
            for k,th in enumerate(threshold):
                tp[n,k] = _true_positive(iou, th)
        fp[n] = n_pred[n] - tp[n]
        fn[n] = n_true[n] - tp[n]
        ap[n] = tp[n] / (tp[n] + fp[n] + fn[n])  
        
    if not_list:
        ap, tp, fp, fn = ap[0], tp[0], fp[0], fn[0]
    return ap, tp, fp, fn

@jit(nopython=True)
def _label_overlap(x, y):
    """ fast function to get pixel overlaps between masks in x and y 
    
    Parameters
    ------------

    x: ND-array, int
        where 0=NO masks; 1,2... are mask labels
    y: ND-array, int
        where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    overlap: ND-array, int
        matrix of pixel overlaps of size [x.max()+1, y.max()+1]
    
    """
    # put label arrays into standard form then flatten them 
    x = x.ravel()
    y = y.ravel()
    
    # preallocate a 'contact map' matrix
    overlap = np.zeros((1+x.max(),1+y.max()), dtype=np.uint)
    
    # loop over the labels in x and add to the corresponding
    # overlap entry. If label A in x and label B in y share P
    # pixels, then the resulting overlap is P
    # len(x)=len(y), the number of pixels in the whole image 
    for i in range(len(x)):
        overlap[x[i],y[i]] += 1
    return overlap

def _intersection_over_union(masks_true, masks_pred):
    """ intersection over union of all mask pairs
    
    Parameters
    ------------
    
    masks_true: ND-array, int 
        ground truth masks, where 0=NO masks; 1,2... are mask labels
    masks_pred: ND-array, int
        predicted masks, where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    iou: ND-array, float
        matrix of IOU pairs of size [x.max()+1, y.max()+1]
    
    ------------
    How it works:
        The overlap matrix is a lookup table of the area of intersection
        between each set of labels (true and predicted). The true labels
        are taken to be along axis 0, and the predicted labels are taken 
        to be along axis 1. The sum of the overlaps along axis 0 is thus
        an array giving the total overlap of the true labels with each of
        the predicted labels, and likewise the sum over axis 1 is the
        total overlap of the predicted labels with each of the true labels.
        Because the label 0 (background) is included, this sum is guaranteed
        to reconstruct the total area of each label. Adding this row and
        column vectors gives a 2D array with the areas of every label pair
        added together. This is equivalent to the union of the label areas
        except for the duplicated overlap area, so the overlap matrix is
        subtracted to find the union matrix. 

    """
    overlap = _label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou

def _true_positive(iou, th):
    """ true positive at threshold th
    
    Parameters
    ------------

    iou: float, ND-array
        array of IOU pairs
    th: float
        threshold on IOU for positive label

    Returns
    ------------

    tp: float
        number of true positives at threshold
        
    ------------
    How it works:
        (1) Find minimum number of masks
        (2) Define cost matrix; for a given threshold, each element is negative
            the higher the IoU is (perfect IoU is 1, worst is 0). The second term
            gets more negative with higher IoU, but less negative with greater
            n_min (but that's a constant...)
        (3) Solve the linear sum assignment problem. The costs array defines the cost
            of matching a true label with a predicted label, so the problem is to 
            find the set of pairings that minimizes this cost. The scipy.optimize
            function gives the ordered lists of corresponding true and predicted labels. 
        (4) Extract the IoUs fro these parings and then threshold to get a boolean array
            whose sum is the number of true positives that is returned. 

    """
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= th).astype(float) - iou / (2*n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    match_ok = iou[true_ind, pred_ind] >= th
    tp = match_ok.sum()
    return tp

def flow_error(maski, dP_net, use_gpu=False, device=None):
    """ error in flows from predicted masks vs flows predicted by network run on image

    This function serves to benchmark the quality of masks, it works as follows
    1. The predicted masks are used to create a flow diagram
    2. The mask-flows are compared to the flows that the network predicted

    If there is a discrepancy between the flows, it suggests that the mask is incorrect.
    Masks with flow_errors greater than 0.4 are discarded by default. Setting can be
    changed in Cellpose.eval or CellposeModel.eval.

    Parameters
    ------------
    
    maski: ND-array (int) 
        masks produced from running dynamics on dP_net, 
        where 0=NO masks; 1,2... are mask labels
    dP_net: ND-array (float) 
        ND flows where dP_net.shape[1:] = maski.shape

    Returns
    ------------

    flow_errors: float array with length maski.max()
        mean squared error between predicted flows and flows from masks
    dP_masks: ND-array (float)
        ND flows produced from the predicted masks
    
    """
    if dP_net.shape[1:] != maski.shape:
        print('ERROR: net flow is not same size as predicted masks')
        return

    # flows predicted from estimated masks
    dP_masks = dynamics.masks_to_flows(maski, use_gpu=use_gpu, device=device)
    # difference between predicted flows vs mask flows
    flow_errors=np.zeros(maski.max())
    for i in range(dP_masks.shape[0]):
        flow_errors += mean((dP_masks[i] - dP_net[i]/5.)**2, maski,
                            index=np.arange(1, maski.max()+1))

    return flow_errors, dP_masks

def mIoU(pred_mask, mask, smooth=1e-10):
    true_class = pred_mask == 1
    true_label = mask == 1
    if true_label.sum() == 0: #no exist label in this loop
        iou = 0
    else:
        intersect = np.logical_and(true_class, true_label).sum()
        union = np.logical_or(true_class, true_label).sum()

        iou = (intersect + smooth) / (union +smooth)
    return iou

def calculate_iou(m1, m2):
    # Initialize a 2D array to store the IoU values
    iou_values = np.zeros((len(np.unique(m2))-1, len(np.unique(m1))-1))
    # Calculate the IoU for each pair of masks
    for i, unid_i in enumerate(np.unique(m2)[1:]):
        for j, unid_j in enumerate(np.unique(m1)[1:]):
            # Calculate intersection and union
            intersection = np.logical_and(m2 == unid_i, m1 == unid_j).sum()
            union = np.logical_or(m2 == unid_i, m1 == unid_j).sum()

            # Calculate IoU and store it in the array
            if union == 0:
                iou_values[i, j] = 0.0
            else:
                iou_values[i, j] = intersection / union

    return iou_values

def gene_iou(s1, s2):
    # convert lists to sets for easier intersection/union operations
    set1 = set(map(tuple, s1))
    set2 = set(map(tuple, s2))
    
    # calculate intersection and union
    intersection = np.array(list(set1 & set2))
    union = np.array(list(set1 | set2))
    
    # Calculate IoU and store it in the array
    if len(union) == 0:
        iou = 0.0
    else:
        iou = len(intersection) / len(union)
    return iou

def compute_IoU(labels, masks_preds):
    assert len(labels) == len(masks_preds)

    all_cell_ious = []
    for masks_pred, label in zip(masks_preds, labels):
        if label.max() != 0:
            benchmark_unique = np.unique(label)[1:]
            unique = np.unique(masks_pred)[1:]

            for b_unique in benchmark_unique:
                bench_mask = label.copy()
                bench_mask[label != b_unique] = 0
                bench_mask[label == b_unique] = 1
                per_cell_ious = []
                for uni in unique:
                    mask = masks_pred.copy()
                    mask[label != uni] = 0
                    mask[label == uni] = 1
                    per_cell_iou = mIoU(mask, bench_mask)
                    per_cell_ious.append(per_cell_iou)
                
                if per_cell_ious:
                    all_cell_ious.append(np.array(per_cell_ious).max())
                else:
                    all_cell_ious.append(0)
        
    iou = np.mean(np.array(all_cell_ious))
    return iou

def compute_gene_IoU(spots, labels, masks_preds):
    assert len(labels) == len(masks_preds) == len(spots)
    all_gene_ious = []
    for i, (label, spot, pred) in enumerate(zip(labels, spots, masks_preds)):
        benchmark_unique = np.unique(label)[1:]
        unique = np.unique(pred)[1:]

        if label.max() != 0 and len(spot) != 0:
            if pred.max() != 0:
                per_benchcell_ious = calculate_iou(pred, label)
                for i, per_benchcell_iou in enumerate(per_benchcell_ious):
                    if per_benchcell_iou.max() != 0:
                        bench_mask = (label == benchmark_unique[i])
                        bench_spot_mask = bench_mask[spot[:,1], spot[:,0]] != 0
                        spot_InCurrentBenchCell = spot[bench_spot_mask]
                        maxIoU_index = np.argmax(per_benchcell_iou)
                        matched_cellid = unique[maxIoU_index]
                        mask = (pred == matched_cellid)
                        spot_mask = mask[spot[:,1], spot[:,0]] != 0
                        spot_InCurrentCell = spot[spot_mask]
                        if len(spot_InCurrentBenchCell) != 0:
                            IoU = gene_iou(spot_InCurrentBenchCell, spot_InCurrentCell)
                            all_gene_ious.append(IoU)
                    else:
                        bench_mask = (label == benchmark_unique[i])
                        bench_spot_mask = bench_mask[spot[:,1], spot[:,0]] != 0
                        spot_InCurrentBenchCell = spot[bench_spot_mask]
                        if len(spot_InCurrentBenchCell) != 0:
                            all_gene_ious.append(0.0)
            else:
                for i in range(len(benchmark_unique)):
                    bench_mask = (label == benchmark_unique[i])
                    bench_spot_mask = bench_mask[spot[:,1], spot[:,0]] != 0
                    spot_InCurrentBenchCell = spot[bench_spot_mask]
                    if len(spot_InCurrentBenchCell) != 0:
                        all_gene_ious.append(0.0)

    all_gene_ious = np.array(all_gene_ious)
    avg_iou = np.mean(all_gene_ious)
    return avg_iou

def compute_IoU_with_GT(args, N):
    test_dir_list = os.listdir(args.test_dir)
    test_subdir_GT_list = []
    test_subdir_pred_list = []
    for each_test_dir in test_dir_list:
        test_subdir_GT_list.append(os.path.join(args.test_dir, each_test_dir + '/GT'))
        test_subdir_pred_list.append(os.path.join(args.test_dir, each_test_dir + '/visresults_{}th'.format(N)))
    
    test_subdir_GT_list = natsorted(test_subdir_GT_list)
    test_subdir_pred_list = natsorted(test_subdir_pred_list)
    GT_file_paths = []
    pred_file_paths = []
    for each_GT_dir, each_pred_dir in zip(test_subdir_GT_list, test_subdir_pred_list):
        GT_file_paths.extend(natsorted(glob.glob(os.path.join(each_GT_dir, '*.png'))))
        pred_file_paths.extend(natsorted(glob.glob(os.path.join(each_pred_dir, '*_label.png'))))
    
    GT_files = []
    pred_files = []
    for GT_file_path, pred_file_path in zip(GT_file_paths, pred_file_paths):
        GT_files.append(cv2.imread(GT_file_path, -1))
        pred_files.append(cv2.imread(pred_file_path, -1))
    

    assert len(GT_files) == len(pred_files)

    all_cell_ious = []
    for masks_pred, label in zip(pred_files, GT_files):
        if label.max() != 0:
            benchmark_unique = np.unique(label)[1:]
            unique = np.unique(masks_pred)[1:]

            for b_unique in benchmark_unique:
                bench_mask = label.copy()
                bench_mask[label != b_unique] = 0
                bench_mask[label == b_unique] = 1
                per_cell_ious = []
                for uni in unique:
                    mask = masks_pred.copy()
                    mask[masks_pred != uni] = 0
                    mask[masks_pred == uni] = 1
                    per_cell_iou = mIoU(mask, bench_mask)
                    per_cell_ious.append(per_cell_iou)
                
                if per_cell_ious:
                    all_cell_ious.append(np.array(per_cell_ious).max())
                else:
                    all_cell_ious.append(0)
        
    iou = np.mean(np.array(all_cell_ious))
    return iou
    
def compute_gene_IoU_with_GT(args, N):
    test_dir_list = os.listdir(args.test_dir)
    test_subdir_GT_list = []
    test_subdir_spot_list = []
    test_subdir_pred_list = []
    for each_test_dir in test_dir_list:
        test_subdir_GT_list.append(os.path.join(args.test_dir, each_test_dir + '/GT'))
        test_subdir_spot_list.append(os.path.join(args.test_dir, each_test_dir + '/spots'))
        test_subdir_pred_list.append(os.path.join(args.test_dir, each_test_dir + '/visresults_{}th'.format(N)))
    
    test_subdir_GT_list = natsorted(test_subdir_GT_list)
    test_subdir_spot_list = natsorted(test_subdir_spot_list)
    test_subdir_pred_list = natsorted(test_subdir_pred_list)
    
    GT_file_paths = []
    spot_file_paths = []
    pred_file_paths = []
    for each_GT_dir, each_spot_dir, each_pred_dir in zip(test_subdir_GT_list, test_subdir_spot_list, test_subdir_pred_list):
        GT_file_paths.extend(natsorted(glob.glob(os.path.join(each_GT_dir, '*.png'))))
        spot_file_paths.extend(natsorted(glob.glob(os.path.join(each_spot_dir, '*.csv'))))
        pred_file_paths.extend(natsorted(glob.glob(os.path.join(each_pred_dir, '*_label.png'))))
    
    GT_files = []
    spot_files = []
    pred_files = []
    for GT_file_path, spot_file_path, pred_file_path in zip(GT_file_paths, spot_file_paths, pred_file_paths):
        GT_files.append(cv2.imread(GT_file_path, -1))
        pred_files.append(cv2.imread(pred_file_path, -1))
        
        spot_list = []
        with open(spot_file_path, 'r') as f:
            lines = f.readlines()
            lines = lines[1:]
        for line in lines:
            splits = line.split(',')
            spot_list.append([int(float(splits[0])), int(float(splits[1]))])
        spot = np.array(spot_list)
        spot_files.append(spot)

    assert len(GT_files) == len(pred_files) == len(spot_files)

    all_gene_ious = []
    for i, (label, spot, pred) in enumerate(zip(GT_files, spot_files, pred_files)):
        benchmark_unique = np.unique(label)[1:]
        unique = np.unique(pred)[1:]
        
        if label.max() != 0 and len(spot) != 0:
            if pred.max() != 0:
                per_benchcell_ious = calculate_iou(pred, label)
                for i, per_benchcell_iou in enumerate(per_benchcell_ious):
                    if per_benchcell_iou.max() != 0:
                        bench_mask = (label == benchmark_unique[i])
                        bench_spot_mask = bench_mask[spot[:,1], spot[:,0]] != 0
                        spot_InCurrentBenchCell = spot[bench_spot_mask]
                        maxIoU_index = np.argmax(per_benchcell_iou)
                        matched_cellid = unique[maxIoU_index]
                        mask = (pred == matched_cellid)
                        spot_mask = mask[spot[:,1], spot[:,0]] != 0
                        spot_InCurrentCell = spot[spot_mask]
                        if len(spot_InCurrentBenchCell) != 0:
                            IoU = gene_iou(spot_InCurrentBenchCell, spot_InCurrentCell)
                            all_gene_ious.append(IoU)
                    else:
                        bench_mask = (label == benchmark_unique[i])
                        bench_spot_mask = bench_mask[spot[:,1], spot[:,0]] != 0
                        spot_InCurrentBenchCell = spot[bench_spot_mask]
                        if len(spot_InCurrentBenchCell) != 0:
                            all_gene_ious.append(0.0)
            else:
                for i in range(len(benchmark_unique)):
                    bench_mask = (label == benchmark_unique[i])
                    bench_spot_mask = bench_mask[spot[:,1], spot[:,0]] != 0
                    spot_InCurrentBenchCell = spot[bench_spot_mask]
                    if len(spot_InCurrentBenchCell) != 0:
                        all_gene_ious.append(0.0)

    all_gene_ious = np.array(all_gene_ious)
    avg_iou = np.mean(all_gene_ious)
    return avg_iou
