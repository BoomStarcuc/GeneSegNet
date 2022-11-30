import os
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import utils, Gseg_io, transforms

try:
    import matplotlib
    MATPLOTLIB_ENABLED = True 
except:
    MATPLOTLIB_ENABLED = False

try:
    from skimage import color
    from skimage.segmentation import find_boundaries
    SKIMAGE_ENABLED = True 
except:
    SKIMAGE_ENABLED = False

# modified to use sinebow color
def dx_to_circ(dP,transparency=False,mask=None):
    """ dP is 2 x Y x X => 'optic' flow representation 
    
    Parameters
    -------------
    
    dP: 2xLyxLx array
        Flow field components [dy,dx]
        
    transparency: bool, default False
        magnitude of flow controls opacity, not lightness (clear background)
        
    mask: 2D array 
        Multiplies each RGB component to suppress noise
    
    """
    #print("dP:", np.array(dP).shape)
    dP = np.array(dP)
    mag = np.clip(transforms.normalize99(np.sqrt(np.sum(dP**2,axis=0))), 0, 1.)
    angles = np.arctan2(dP[1], dP[0])+np.pi
    a = 2
    r = ((np.cos(angles)+1)/a)
    g = ((np.cos(angles+2*np.pi/3)+1)/a)
    b =((np.cos(angles+4*np.pi/3)+1)/a)
    
    if transparency:
        im = np.stack((r,g,b,mag),axis=-1)
    else:
        im = np.stack((r*mag,b*mag,g*mag),axis=-1)
        
    if mask is not None and transparency and dP.shape[0]<3:
        im[:,:,-1] *= mask
        
    im = (np.clip(im, 0, 1) * 255).astype(np.uint8)
    return im

def show_segmentation(fig, img, maski, flowi, labeli, spot, channels=[0,0], file_name=None):
    """ plot segmentation results
    
    Can save each panel of figure with file_name option. Use channels option if
    img input is not an RGB image with 3 channels.
    
    Parameters
    -------------

    fig: matplotlib.pyplot.figure
        figure in which to make plot

    img: 2D or 3D array
        image input into GeneSegNet

    maski: int, 2D array
        for image k, masks[k] output from GeneSegNet.eval, where 0=NO masks; 1,2,...=mask labels

    flowi: int, 2D array 
        for image k, flows[k][0] output from GeneSegNet.eval (RGB of flows)

    channels: list of int (optional, default [0,0])
        channels used to run GeneSegNet, no need to use if image is RGB

    file_name: str (optional, default None)
        file name of image, if file_name is not None, figure panels are saved
        
    seg_norm: bool (optional, default False)
        improve cell visibility under labels
        

    """
    if not MATPLOTLIB_ENABLED:
        raise ImportError("matplotlib not installed, install with 'pip install matplotlib'")
    
    spot_in = []
    spot_out = []
    for x,y in spot:
        if maski[y,x] != 0:
            spot_in.append([x,y])
        else:
            spot_out.append([x,y])
    
    spot_in = np.array(spot_in)
    spot_out = np.array(spot_out)

    label_in = []
    label_out = []
    for x,y in spot:
        if labeli[y,x] != 0:
            label_in.append([x,y])
        else:
            label_out.append([x,y])
    
    label_in = np.array(label_in)
    label_out = np.array(label_out)

    img0 = img.copy()

    if img0.shape[0] < 4:
        img0 = np.transpose(img0, (1,2,0))
    if img0.shape[-1] < 3 or img0.ndim < 3:
        img0 = image_to_rgb(img0, channels=channels)
    else:
        if img0.max()<=50.0:
            img0 = np.uint8(np.clip(img0*255, 0, 1))
    
    ax = fig.add_subplot(1,6,1)
    ax.imshow(img0)
    ax.set_title('input image')
    ax.axis('off')
    
    outlines = utils.masks_to_outlines(maski)
    overlay = mask_overlay(img0, maski)

    outlines_gt = utils.masks_to_outlines(labeli)
    overlay_gt = mask_overlay(img0, labeli)

    ax = fig.add_subplot(1,6,2)
    outX, outY = np.nonzero(outlines)
    imgout= img0.copy()

    if len(spot_in) != 0:
        ax.scatter(spot_in[:,0],spot_in[:,1], marker='o', c='c', s=1)
    
    if len(spot_out) != 0:
        ax.scatter(spot_out[:,0], spot_out[:,1], marker='o',c='y', s=1)
    
    ax.scatter(outY, outX, marker = ',' , c='r', lw=0, s=0.1)
    ax.imshow(imgout)
    ax.set_title('predicted outlines')
    ax.axis('off')

    ax = fig.add_subplot(1,6,3)
    ax.imshow(overlay)
    ax.set_title('predicted masks')
    ax.axis('off')

    ax = fig.add_subplot(1,6,4)
    outX_gt, outY_gt = np.nonzero(outlines_gt)
    imgout = img0.copy()

    if len(label_in) != 0:
        ax.scatter(label_in[:,0],label_in[:,1], marker='o', c='c', s=1)
    
    if len(label_out) != 0:
        ax.scatter(label_out[:,0],label_out[:,1], marker='o', c='y', s=1)
    
    ax.scatter(outY_gt, outX_gt, marker = ',' , c='r', lw=0, s=0.1)
    ax.imshow(imgout)
    ax.set_title('pciSeq outlines')
    ax.axis('off')

    ax = fig.add_subplot(1,6,5)
    ax.imshow(overlay_gt)
    ax.set_title('pciSeq masks')
    ax.axis('off')

    ax = fig.add_subplot(1,6,6)
    ax.imshow(flowi)
    ax.set_title('predicted offset')
    ax.axis('off')

    if file_name is not None:
        save_path = os.path.splitext(file_name)[0]
        Gseg_io.imsave(save_path + '_overlay.jpg', overlay)
        Gseg_io.imsave(save_path + '_outlines.jpg', imgout)
        Gseg_io.imsave(save_path + '_flows.jpg', flowi)

def mask_rgb(masks, colors=None):
    """ masks in random rgb colors

    Parameters
    ----------------

    masks: int, 2D array
        masks where 0=NO masks; 1,2,...=mask labels

    colors: int, 2D array (optional, default None)
        size [nmasks x 3], each entry is a color in 0-255 range

    Returns
    ----------------

    RGB: uint8, 3D array
        array of masks overlaid on grayscale image

    """
    if colors is not None:
        if colors.max()>1:
            colors = np.float32(colors)
            colors /= 255
        colors = utils.rgb_to_hsv(colors)
    
    HSV = np.zeros((masks.shape[0], masks.shape[1], 3), np.float32)
    HSV[:,:,2] = 1.0
    for n in range(int(masks.max())):
        ipix = (masks==n+1).nonzero()
        if colors is None:
            HSV[ipix[0],ipix[1],0] = np.random.rand()
        else:
            HSV[ipix[0],ipix[1],0] = colors[n,0]
        HSV[ipix[0],ipix[1],1] = np.random.rand()*0.5+0.5
        HSV[ipix[0],ipix[1],2] = np.random.rand()*0.5+0.5
    RGB = (utils.hsv_to_rgb(HSV) * 255).astype(np.uint8)
    return RGB

def mask_overlay(img, masks, colors=None):
    """ overlay masks on image (set image to grayscale)

    Parameters
    ----------------

    img: int or float, 2D or 3D array
        img is of size [Ly x Lx (x nchan)]

    masks: int, 2D array
        masks where 0=NO masks; 1,2,...=mask labels

    colors: int, 2D array (optional, default None)
        size [nmasks x 3], each entry is a color in 0-255 range

    Returns
    ----------------

    RGB: uint8, 3D array
        array of masks overlaid on grayscale image

    """
    if colors is not None:
        if colors.max()>1:
            colors = np.float32(colors)
            colors /= 255
        colors = utils.rgb_to_hsv(colors)
    if img.ndim>2:
        img = img.astype(np.float32).mean(axis=-1)
    else:
        img = img.astype(np.float32)
    
    HSV = np.zeros((img.shape[0], img.shape[1], 3), np.float32)
    HSV[:,:,2] = np.clip((img / 255. if img.max() > 1 else img) * 1.5, 0, 1)
    hues = np.linspace(0, 1, masks.max()+1)[np.random.permutation(masks.max())]
    for n in range(int(masks.max())):
        ipix = (masks==n+1).nonzero()
        if colors is None:
            HSV[ipix[0],ipix[1],0] = hues[n]
        else:
            HSV[ipix[0],ipix[1],0] = colors[n,0]
        HSV[ipix[0],ipix[1],1] = 1.0
    RGB = (utils.hsv_to_rgb(HSV) * 255).astype(np.uint8)
    return RGB

def image_to_rgb(img0, channels=[0,0]):
    """ image is 2 x Ly x Lx or Ly x Lx x 2 - change to RGB Ly x Lx x 3 """
    img = img0.copy()
    img = img.astype(np.float32)
    if img.ndim<3:
        img = img[:,:,np.newaxis]
    if img.shape[0]<5:
        img = np.transpose(img, (1,2,0))
    if channels[0]==0:
        img = img.mean(axis=-1)[:,:,np.newaxis]
    for i in range(img.shape[-1]):
        if np.ptp(img[:,:,i])>0:
            img[:,:,i] = np.clip(transforms.normalize99(img[:,:,i]), 0, 1)
            img[:,:,i] = np.clip(img[:,:,i], 0, 1)
    img *= 255
    img = np.uint8(img)
    RGB = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    if img.shape[-1]==1:
        RGB = np.tile(img,(1,1,3))
    else:
        RGB[:,:,channels[0]-1] = img[:,:,0]
        if channels[1] > 0:
            RGB[:,:,channels[1]-1] = img[:,:,1]
    return RGB

def interesting_patch(mask, bsize=130):
    """ get patch of size bsize x bsize with most masks """
    Ly,Lx = mask.shape
    m = np.float32(mask>0)
    m = gaussian_filter(m, bsize/2)
    y,x = np.unravel_index(np.argmax(m), m.shape)
    ycent = max(bsize//2, min(y, Ly-bsize//2))
    xcent = max(bsize//2, min(x, Lx-bsize//2))
    patch = [np.arange(ycent-bsize//2, ycent+bsize//2, 1, int),
             np.arange(xcent-bsize//2, xcent+bsize//2, 1, int)]
    return patch

def disk(med, r, Ly, Lx):
    """ returns pixels of disk with radius r and center med """
    yy, xx = np.meshgrid(np.arange(0,Ly,1,int), np.arange(0,Lx,1,int),
                         indexing='ij')
    inds = ((yy-med[0])**2 + (xx-med[1])**2)**0.5 <= r
    y = yy[inds].flatten()
    x = xx[inds].flatten()
    return y,x

def outline_view(img0,maski,color=[1,0,0], mode='inner'):
    """
    Generates a red outline overlay onto image. 
    """
#     img0 = utils.rescale(img0)
    if len(img0.shape)<3:
#         img0 = image_to_rgb(img0) broken, transposing some images...
        img0 = np.stack([img0]*3,axis=-1)
    
    if SKIMAGE_ENABLED:
        outlines = find_boundaries(maski,mode=mode) #not using masks_to_outlines as that gives border 'outlines'
    else:
        outlines = utils.masks_to_outlines(maski,mode=mode) #not using masks_to_outlines as that gives border 'outlines'
    outY, outX = np.nonzero(outlines)
    imgout = img0.copy()
#     imgout[outY, outX] = np.array([255,0,0]) #pure red
    imgout[outY, outX] = np.array(color)

    return imgout
