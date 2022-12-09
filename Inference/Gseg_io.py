import os, datetime, gc, warnings, glob, shutil
from natsort import natsorted
import numpy as np
import cv2
import tifffile
import logging, pathlib, sys
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import fastremap
from PIL import Image

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except:
    MATPLOTLIB = False
  
io_logger = logging.getLogger(__name__)

def logger_setup():
    cp_dir = pathlib.Path.home().joinpath('.GeneSegNet')
    cp_dir.mkdir(exist_ok=True)
    log_file = cp_dir.joinpath('GeneSegNet.log')
    try:
        log_file.unlink()
    except:
        print('creating new log file')
    logging.basicConfig(
                    level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler(sys.stdout)
                    ]
                )
    logger = logging.getLogger(__name__)
    logger.info(f'WRITING LOG OUTPUT TO {log_file}')

    return logger, log_file

import utils, plot, transforms

# helper function to check for a path; if it doesn't exist, make it 
def check_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def outlines_to_text(base, outlines):
    with open(base + '_cp_outlines.txt', 'w') as f:
        for o in outlines:
            xy = list(o.flatten())
            xy_str = ','.join(map(str, xy))
            f.write(xy_str)
            f.write('\n')

def imread(filename, img_type):
    ext = os.path.splitext(filename)[-1]
    if ext == '.tif' or ext =='.tiff':
        with tifffile.TiffFile(filename) as tif:
            ltif = len(tif.pages)
            try:
                full_shape = tif.shaped_metadata[0]['shape']
            except:
                try:
                    page = tif.series[0][0]
                    full_shape = tif.series[0].shape
                except:
                    ltif = 0
            if ltif < 10:
                img = tif.asarray()
            else:
                page = tif.series[0][0]
                shape, dtype = page.shape, page.dtype
                ltif = int(np.prod(full_shape) / np.prod(shape))
                io_logger.info(f'reading tiff with {ltif} planes')
                img = np.zeros((ltif, *shape), dtype=dtype)
                for i,page in enumerate(tqdm(tif.series[0])):
                    img[i] = page.asarray()
                img = img.reshape(full_shape)            
        return img
    elif ext != '.npy':
        try:
            if img_type == 'image':
                img = cv2.imread(filename)
                img = img[:,:,0]
            elif img_type == 'label':
                img = cv2.imread(filename, -1)
            else:
                img = cv2.imread(filename, 0)
            
            return img
        except Exception as e:
            io_logger.critical('ERROR: could not read file, %s'%e)
            return None
    else:
        try:
            dat = np.load(filename, allow_pickle=True).item()
            masks = dat['masks']
            return masks
        except Exception as e:
            io_logger.critical('ERROR: could not read masks from file, %s'%e)
            return None

def imsave(filename, arr):
    ext = os.path.splitext(filename)[-1]
    if ext== '.tif' or ext=='.tiff':
        tifffile.imsave(filename, arr)
    else:
        if len(arr.shape)>2:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        cv2.imwrite(filename, arr)

def get_image_files(folder, img_filter='_image'):
    """ find all images in a folder """
    image_names = []
    image_names.extend(glob.glob(os.path.join(folder, "images") + '/*%s.png'%img_filter))
    image_names.extend(glob.glob(os.path.join(folder, "images") + '/*%s.jpg'%img_filter))
    image_names.extend(glob.glob(os.path.join(folder, "images") + '/*%s.jpeg'%img_filter))
    image_names.extend(glob.glob(os.path.join(folder, "images") + '/*%s.tif'%img_filter))
    image_names.extend(glob.glob(os.path.join(folder, "images") + '/*%s.tiff'%img_filter))
    image_names = natsorted(image_names)

    if len(image_names)==0:
        raise ValueError('ERROR: no images in --dir folder')
    
    return image_names

def get_label_files(folder, N, mask_filter='_label', foldername = 'newlabels'):
    label_names = []
    
    label_path = os.path.join(folder, '{}'.format(foldername))
    if N == 1 and os.path.exists(label_path):
        shutil.rmtree(label_path)
    
    # print("@@@@@@@@@@@@@@@@@@label_path@@@@@@@@@@@@@@@@@@@@@@@", label_path)
    if not os.path.exists(label_path) or N == 0:
        print("#################enter label @@@@@@@@@@@@@@@@@@@@@@@@@")
        label_names.extend(glob.glob(os.path.join(folder, "labels") + '/*%s.png'%mask_filter))
        label_names.extend(glob.glob(os.path.join(folder, "labels") + '/*%s.jpg'%mask_filter))
        label_names.extend(glob.glob(os.path.join(folder, "labels") + '/*%s.jpeg'%mask_filter))
        label_names.extend(glob.glob(os.path.join(folder, "labels") + '/*%s.tif'%mask_filter))
        label_names.extend(glob.glob(os.path.join(folder, "labels") + '/*%s.tiff'%mask_filter))
    else:
        print("#################enter output @@@@@@@@@@@@@@@@@@@@@@@@@")
        label_names.extend(glob.glob(label_path + '/*%s.png'%mask_filter))
        label_names.extend(glob.glob(label_path + '/*%s.jpg'%mask_filter))
        label_names.extend(glob.glob(label_path + '/*%s.jpeg'%mask_filter))
        label_names.extend(glob.glob(label_path + '/*%s.tif'%mask_filter))
        label_names.extend(glob.glob(label_path + '/*%s.tiff'%mask_filter))
    
    label_names = natsorted(label_names)
    # print("label_names:", label_names, len(label_names))
    label_names0 = [os.path.splitext(label_names[n])[0] for n in range(len(label_names))]
    
    # check for flows
    flow_names = [label_names0[n] + '_flows.tif' for n in range(len(label_names0))]
    if not all([os.path.exists(flow) for flow in flow_names]):
        io_logger.info('not all flows are present, running flow generation for all images')
        flow_names = None

    if not all([os.path.exists(label) for label in label_names]):
        raise ValueError('labels not provided for all images in train and/or test set')

    return label_names, flow_names

def get_heatmap_files(folder, heatmap_filter='_gaumap_all'):
    heatmap_all_names = []
    heatmap_all_names.extend(glob.glob(os.path.join(os.path.join(folder, "HeatMaps"), "HeatMap_all") + '/*%s.png'% heatmap_filter))
    heatmap_all_names.extend(glob.glob(os.path.join(os.path.join(folder, "HeatMaps"), "HeatMap_all") + '/*%s.jpg'% heatmap_filter))
    heatmap_all_names.extend(glob.glob(os.path.join(os.path.join(folder, "HeatMaps"), "HeatMap_all") + '/*%s.jpeg'% heatmap_filter))
    heatmap_all_names.extend(glob.glob(os.path.join(os.path.join(folder, "HeatMaps"), "HeatMap_all")+ '/*%s.tif'% heatmap_filter))
    heatmap_all_names.extend(glob.glob(os.path.join(os.path.join(folder, "HeatMaps"), "HeatMap_all") + '/*%s.tiff'% heatmap_filter))
    heatmap_all_names = natsorted(heatmap_all_names)

    if len(heatmap_all_names)==0:
        raise ValueError('ERROR: no gaussian maps')

    return heatmap_all_names

def get_spot_files(dir):
    spot_names = []
    spot_names.extend(glob.glob(os.path.join(os.path.join(dir, 'spots/'), '*.csv')))
    spot_names = natsorted(spot_names)

    return spot_names

def load_images_labels_heatmap_spot(dir, N, mask_filter='_masks', image_filter='_label', heatmap_filter='_gaumap_all', foldername = 'newlabels'):
    image_names = get_image_files(dir, image_filter)
    label_names, flow_names = get_label_files(dir, N, mask_filter, foldername)
    heatmap_names = get_heatmap_files(dir, heatmap_filter)
    spots_names = get_spot_files(dir)

    if flow_names is None:
        assert len(image_names) == len(label_names) == len(heatmap_names) == len(spots_names)
    else:
        assert len(image_names) == len(label_names) == len(flow_names) == len(heatmap_names) == len(spots_names)

    nimg = len(image_names)
    
    images = []
    labels = []
    heatmaps = []
    spots = []
    k = 0
    for n in range(nimg):
        image = imread(image_names[n], 'image')
        label = imread(label_names[n], 'label')
        heatmap = imread(heatmap_names[n], 'heatmap')

        spot_list = []
        with open(spots_names[n], 'r') as f:
            lines = f.readlines()
            lines = lines[1:]

        for line in lines:
            splits = line.split(',')
            spot_list.append([int(float(splits[0])), int(float(splits[1]))])

        spot = np.array(spot_list)

        if flow_names is not None:
            flow = imread(flow_names[n], 'flow')
            if flow.shape[0]<4:
                label = np.concatenate((label[np.newaxis,:,:], flow), axis=0) 
            else:
                label = flow
        
        images.append(image)
        labels.append(label)
        heatmaps.append(heatmap)
        spots.append(spot)

        k+=1
    
    io_logger.info(f'{k} / {nimg} images in {dir} folder have labels')
    return images, labels, heatmaps, spots, label_names

def load_train_test_data(train_dir, N, test_dir=None, image_filter='_image', mask_filter='_label', heatmap_filter='_gaumap_all', foldername = 'newlabels'):
    train_dir_list = os.listdir(train_dir)
    train_subdir_list = []
    for each_train_dir in train_dir_list:
        train_subdir_list.append(os.path.join(train_dir, each_train_dir))
    
    train_subdir_list = natsorted(train_subdir_list)

    #train data
    images_list = []
    labels_list = []
    heatmaps_list = []
    spots_list = []
    label_names_list = []
    
    for each_train_dir in train_subdir_list:
        images, labels, heatmaps, spots, label_names = load_images_labels_heatmap_spot(each_train_dir, N, mask_filter, 
                                                                            image_filter, heatmap_filter, 
                                                                            foldername)
        images_list.append(np.array(images))
        labels_list.append(np.array(labels))
        heatmaps_list.append(np.array(heatmaps))
        label_names_list.append(np.array(label_names))  
        spots_list.extend(spots)

    images = np.concatenate(images_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    heatmaps = np.concatenate(heatmaps_list, axis=0)
    spots = spots_list
    label_names = np.concatenate(label_names_list, axis=0)
                   
    # testing data
    test_images, test_labels, test_heatmaps, test_spots, test_label_names = None, None, None, None, None
    if test_dir is not None:
        test_dir_list = os.listdir(test_dir)
        test_subdir_list = []
        for each_test_dir in test_dir_list:
            test_subdir_list.append(os.path.join(test_dir, each_test_dir))

        test_subdir_list = natsorted(test_subdir_list)

        test_images_list = []
        test_labels_list = []
        test_heatmaps_list = []
        test_spots_list = []
        test_label_names_list = []
        for each_test_dir in test_subdir_list:
            test_images, test_labels, test_heatmaps, test_spots, test_label_names = load_images_labels_heatmap_spot(each_test_dir, N, mask_filter, 
                                                                                                    image_filter, heatmap_filter, 
                                                                                                    foldername)

            test_images_list.append(np.array(test_images))
            test_labels_list.append(np.array(test_labels))
            test_heatmaps_list.append(np.array(test_heatmaps))
            test_label_names_list.append(np.array(test_label_names))
            test_spots_list.extend(test_spots)

        test_images = np.concatenate(test_images_list, axis=0)
        test_labels = np.concatenate(test_labels_list, axis=0)
        test_heatmaps = np.concatenate(test_heatmaps_list, axis=0)
        test_spots = test_spots_list
        test_label_names = np.concatenate(test_label_names_list, axis=0)

    return images, labels, heatmaps, spots, label_names, test_images, test_labels, test_heatmaps, test_spots, test_label_names

# Now saves flows, masks, etc. to separate folders.
def save_masks(images, masks, flows, label, spot, file_names, png=True, tif=False, foldername = 'newlabels', channels=[0,0],
               suffix='',save_flows=False, save_outlines=False, save_ncolor=False, 
               dir_above=False, in_folders=False, savedir=None, save_txt=True):
    """ save masks + nicely plotted segmentation image to png and/or tiff

    if png, masks[k] for images[k] are saved to file_names[k]+'_cp_masks.png'

    if tif, masks[k] for images[k] are saved to file_names[k]+'_cp_masks.tif'

    if png and matplotlib installed, full segmentation figure is saved to file_names[k]+'_cp.png'

    only tif option works for 3D data
    
    Parameters
    -------------

    images: (list of) 2D, 3D or 4D arrays
        images input into GeneSegNet

    masks: (list of) 2D arrays, int
        masks output from GeneSegNet.eval, where 0=NO masks; 1,2,...=mask labels

    flows: (list of) list of ND arrays 
        flows output from GeneSegNet.eval

    file_names: (list of) str
        names of files of images
        
    savedir: str
        absolute path where images will be saved. Default is none (saves to image directory)
    
    save_flows, save_outlines, save_ncolor, save_txt: bool
        Can choose which outputs/views to save.
        ncolor is a 4 (or 5, if 4 takes too long) index version of the labels that
        is way easier to visualize than having hundreds of unique colors that may
        be similar and touch. Any color map can be applied to it (0,1,2,3,4,...).
    
    """

    if isinstance(masks, list):
        for image, mask, flow, file_name in zip(images, masks, flows, file_names):
            save_masks(image, mask, flow, file_name, png=png, tif=tif, suffix=suffix,dir_above=dir_above,
                       save_flows=save_flows,save_outlines=save_outlines,save_ncolor=save_ncolor,
                       savedir=savedir,save_txt=save_txt,in_folders=in_folders)
        return
    
    if masks.ndim > 2 and not tif:
        raise ValueError('cannot save 3D outputs as PNG, use tif option instead')
    
    if savedir is None: 
        savedir = str(Path(file_names).parent.parent.absolute()) + '/{}'.format(foldername)
    
    check_dir(savedir) 
            
    basename = os.path.splitext(os.path.basename(file_names))[0]
    if in_folders:
        maskdir = os.path.join(savedir,'masks')
        outlinedir = os.path.join(savedir,'outlines')
        txtdir = os.path.join(savedir,'txt_outlines')
        ncolordir = os.path.join(savedir,'ncolor_masks')
        flowdir = os.path.join(savedir,'flows')
    else:
        maskdir = savedir
        outlinedir = savedir
        txtdir = savedir
        ncolordir = savedir
        flowdir = savedir
        
    check_dir(maskdir) 
    exts = []
    if masks.ndim > 2:
        png = False
        tif = True
    
    if png:    
        if masks.max() < 2**16:
            masks = masks.astype(np.uint16) 
            exts.append('.png')
        else:
            png = False 
            tif = True
            io_logger.warning('found more than 65535 masks in each image, cannot save PNG, saving as TIF')
    if tif:
        exts.append('.tif')

    # save masks
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ext in exts:
            label_remap = fastremap.renumber(label, in_place=True)[0]
            plt.imshow(label_remap)
            plt.axis('off')
            plt.savefig(os.path.join(maskdir,basename + '_cp_Gt' + suffix + ext), bbox_inches='tight', pad_inches = 0 )
            plt.clf()

            im = Image.fromarray(masks)
            im_path = os.path.join(maskdir,basename[:-5] + 'label' + suffix + ext)
            im.save(im_path)

            plt.imshow(masks)
            plt.axis('off')
            plt.savefig(os.path.join(maskdir,basename + '_cp_mask' + suffix + ext), bbox_inches='tight', pad_inches = 0 )

            plt.imshow(flows[2])
            plt.axis('off')
            plt.savefig(os.path.join(maskdir,basename + '_confidencemap' + suffix + ext), bbox_inches='tight', pad_inches = 0 )
            plt.clf()

            plt.imshow(flows[3])
            plt.axis('off')
            plt.savefig(os.path.join(maskdir,basename + '_centermap' + suffix + ext), bbox_inches='tight', pad_inches = 0 )
            plt.clf()
            
    if png and MATPLOTLIB:
        img = images.copy()
        if img.ndim<3:
            img = img[:,:,np.newaxis]
        elif img.shape[0]<8:
            np.transpose(img, (1,2,0))
        
        # print("Gseg_io_img:", img.shape)
        fig = plt.figure(figsize=(12,3))
        plot.show_segmentation(fig, img, masks, flows[0], label, spot)
        fig.savefig(os.path.join(savedir,basename + '_cp_output' + suffix + '.png'), dpi=300)
        plt.close(fig)

    # ImageJ txt outline files 
    if masks.ndim < 3 and save_txt:
        check_dir(txtdir)
        outlines = utils.outlines_list(masks)
        outlines_to_text(os.path.join(txtdir,basename), outlines)
    
    # RGB outline images
    if masks.ndim < 3 and save_outlines: 
        check_dir(outlinedir) 
        outlines = utils.masks_to_outlines(masks)
        outX, outY = np.nonzero(outlines)
        img0 = transforms.normalize99(images)
        if img0.shape[0] < 4:
            img0 = np.transpose(img0, (1,2,0))
        if img0.shape[-1] < 3 or img0.ndim < 3:
            img0 = plot.image_to_rgb(img0, channels=channels)
        else:
            if img0.max()<=50.0:
                img0 = np.uint8(np.clip(img0*255, 0, 1))
        imgout= img0.copy()
        imgout[outX, outY] = np.array([255,0,0]) #pure red 
        imsave(os.path.join(outlinedir, basename + '_outlines' + suffix + '.png'),  imgout)
    
    # save RGB flow picture
    if masks.ndim < 3 and save_flows:
        check_dir(flowdir)
        imsave(os.path.join(flowdir, basename + '_flows' + suffix + '.tif'), (flows[0]*(2**16 - 1)).astype(np.uint16))
        #save full flow data
        imsave(os.path.join(flowdir, basename + '_dP' + suffix + '.tif'), flows[1]) 