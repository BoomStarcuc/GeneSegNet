import os, sys, time, shutil, tempfile, datetime, pathlib, subprocess
from pathlib import Path
import numpy as np
from tqdm import trange, tqdm
from urllib.parse import urlparse

import logging
models_logger = logging.getLogger(__name__)

import transforms, dynamics, utils, plot
from core import UnetModel, assign_device, check_mkl, parse_model_string
import matplotlib.pyplot as plt
import cv2

class GeneSegModel(UnetModel):

    def __init__(self, gpu=False, pretrained_model=False, 
                    model_type=None, net_avg=False,
                    diam_mean=30., device=None,
                    residual_on=True, style_on=True, concatenation=False,
                    nchan=2):
        self.torch = True
        if isinstance(pretrained_model, np.ndarray):
            pretrained_model = list(pretrained_model)
        elif isinstance(pretrained_model, str):
            pretrained_model = [pretrained_model]
    
        self.diam_mean = diam_mean
        # builtin = True
        builtin = False
        if pretrained_model:
            pretrained_model_string = pretrained_model[0]
            params = parse_model_string(pretrained_model_string)
            if params is not None:
                _, residual_on, style_on, concatenation = params 
            models_logger.info(f'>>>> loading model {pretrained_model_string}')
            
        # initialize network
        super().__init__(gpu=gpu, pretrained_model=False,
                         diam_mean=self.diam_mean, net_avg=net_avg, device=device,
                         residual_on=residual_on, style_on=style_on, concatenation=concatenation,
                        nchan=nchan)

        self.unet = False
        self.pretrained_model = pretrained_model
        if self.pretrained_model:
            self.net.load_model(self.pretrained_model[0], cpu=(not self.gpu))
            self.diam_mean = self.net.diam_mean.data.cpu().numpy()[0]
            self.diam_labels = self.net.diam_labels.data.cpu().numpy()[0]
            models_logger.info(f'>>>> model diam_mean = {self.diam_mean: .3f} (ROIs rescaled to this size during training)')
            if not builtin:
                models_logger.info(f'>>>> model diam_labels = {self.diam_labels: .3f} (mean diameter of training ROIs)')
        
        ostr = ['off', 'on']
        self.net_type = 'GeneSegNet_residual_{}_style_{}_concatenation_{}'.format(ostr[residual_on],
                                                                                   ostr[style_on],
                                                                                   ostr[concatenation]
                                                                                 ) 
    
    def eval(self, x, batch_size=8, channels=None, channel_axis=None, 
             z_axis=None, normalize=True, invert=False, 
             rescale=None, diameter=None, do_3D=False, anisotropy=None, net_avg=False, 
             augment=False, tile=True, tile_overlap=0.1,
             resample=True, interp=True,
             flow_threshold=0.4, cellprob_threshold=0.0,
             compute_masks=True, min_size=300, stitch_threshold=0.0, progress=None,  
             loop_run=False, model_loaded=False):

        if isinstance(x, list) or x.squeeze().ndim==5:
            masks, styles, offsetmaps, confimaps, centermaps = [], [], [], [], []
            tqdm_out = utils.TqdmToLogger(models_logger, level=logging.INFO)
            nimg = len(x)
            iterator = trange(nimg, file=tqdm_out) if nimg>1 else range(nimg)
            for i in iterator:
                maski, stylei, offsetmapi, confimapi, centermapi = self.eval(x[i], 
                                                                            batch_size=batch_size, 
                                                                            channels=channels[i] if (len(channels)==len(x) and 
                                                                                                    (isinstance(channels[i], list) or isinstance(channels[i], np.ndarray)) and
                                                                                                    len(channels[i])==2) else channels, 
                                                                            channel_axis=channel_axis, 
                                                                            z_axis=z_axis, 
                                                                            normalize=normalize, 
                                                                            invert=invert, 
                                                                            rescale=rescale[i] if isinstance(rescale, list) or isinstance(rescale, np.ndarray) else rescale,
                                                                            diameter=diameter[i] if isinstance(diameter, list) or isinstance(diameter, np.ndarray) else diameter, 
                                                                            do_3D=do_3D, 
                                                                            anisotropy=anisotropy, 
                                                                            net_avg=net_avg, 
                                                                            augment=augment, 
                                                                            tile=tile, 
                                                                            tile_overlap=tile_overlap,
                                                                            resample=resample, 
                                                                            interp=interp,
                                                                            flow_threshold=flow_threshold,
                                                                            cellprob_threshold=cellprob_threshold, 
                                                                            compute_masks=compute_masks, 
                                                                            min_size=min_size, 
                                                                            stitch_threshold=stitch_threshold, 
                                                                            progress=progress,
                                                                            loop_run=(i>0),
                                                                            model_loaded=model_loaded)
                masks.append(maski)
                offsetmaps.append(offsetmapi)
                confimaps.append(confimapi)
                centermaps.append(centermapi)
                styles.append(stylei)
            return masks, offsetmaps, confimaps, centermaps, styles
        
        else:
            if not model_loaded and (isinstance(self.pretrained_model, list) and not net_avg and not loop_run):
                self.net.load_model(self.pretrained_model[0], cpu=(not self.gpu))

            x = transforms.convert_image(x, channels, channel_axis=channel_axis, z_axis=z_axis,
                                         do_3D=(do_3D or stitch_threshold>0), 
                                         normalize=False, invert=False, nchan=self.nchan)

            if x.ndim < 4:
                x = x[np.newaxis,...]

            self.batch_size = batch_size
            if diameter is not None and diameter > 0:
                rescale = self.diam_mean / diameter
            elif rescale is None:
                diameter = self.diam_labels
                rescale = self.diam_mean / diameter

            masks, styles, offsetmap, confimap, centermap = self._run_cp(x, 
                                                          compute_masks=compute_masks,
                                                          normalize=normalize,
                                                          invert=invert,
                                                          rescale=rescale, 
                                                          net_avg=net_avg, 
                                                          resample=resample,
                                                          augment=augment, 
                                                          tile=tile, 
                                                          tile_overlap=tile_overlap,
                                                          flow_threshold=flow_threshold,
                                                          cellprob_threshold=cellprob_threshold, 
                                                          interp=interp,
                                                          min_size=min_size, 
                                                          do_3D=do_3D, 
                                                          anisotropy=anisotropy,
                                                          stitch_threshold=stitch_threshold,
                                                          )
            
            flows = [plot.dx_to_circ(offsetmap), offsetmap, confimap, centermap]
            return masks, flows, styles

    def _run_cp(self, x, compute_masks=True, normalize=True, invert=False,
                rescale=1.0, net_avg=False, resample=True,
                augment=False, tile=True, tile_overlap=0.1,
                cellprob_threshold=0.0, 
                flow_threshold=0.4, min_size=300,
                interp=True, anisotropy=1.0, do_3D=False, stitch_threshold=0.0,
                ):

        tic = time.time()
        shape = x.shape
        nimg = shape[0]       

        tqdm_out = utils.TqdmToLogger(models_logger, level=logging.INFO)
        iterator = trange(nimg, file=tqdm_out) if nimg>1 else range(nimg)
        styles = np.zeros((nimg, self.nbase[-1]), np.float32)

        if resample:
            offsetmap = np.zeros((2, nimg, shape[1], shape[2]), np.float32)
            confimap = np.zeros((nimg, shape[1], shape[2]), np.float32)
            centermap = np.zeros((nimg, shape[1], shape[2]), np.float32)
        else:
            offsetmap = np.zeros((2, nimg, int(shape[1]*rescale), int(shape[2]*rescale)), np.float32)
            confimap = np.zeros((nimg, int(shape[1]*rescale), int(shape[2]*rescale)), np.float32)
            centermap = np.zeros((nimg, shape[1], shape[2]), np.float32)
            
        for i in iterator:
            img = np.asarray(x[i])
            if normalize or invert:
                img = transforms.normalize_img(img, invert=invert)
            if rescale != 1.0:
                img = transforms.resize_image(img, rsz=rescale)
            
            yf, style = self._run_nets(img, net_avg=net_avg,
                                        augment=augment, tile=tile,
                                        tile_overlap=tile_overlap)
            
            if resample:
                yf = transforms.resize_image(yf, shape[1], shape[2])

            confimap[i] = yf[:,:,2]
            offsetmap[:, i] = yf[:,:,:2].transpose((2,0,1))
            centermap[i] = yf[:,:,3]

            styles[i] = style
        del yf, style
        
        styles = styles.squeeze()
        
        net_time = time.time() - tic
        if nimg > 1:
            models_logger.info('network run in %2.2fs'%(net_time))

        if compute_masks:
            tic=time.time()
            masks = []
            resize = [shape[1], shape[2]] if not resample else None
            
            for i in iterator:
                outputs = dynamics.compute_masks(offsetmap[:,i], centermap[i], confimap[i], confidence_threshold=cellprob_threshold,
                                                        flow_threshold=flow_threshold, interp=interp, resize=resize,
                                                        use_gpu=self.gpu, device=self.device)
                masks.append(outputs)
                
            masks = np.array(masks)
            flow_time = time.time() - tic
            
            if nimg > 1:
                models_logger.info('masks created in %2.2fs'%(flow_time))
            masks, offsetmap, confimap, centermap = masks.squeeze(), offsetmap.squeeze(), confimap.squeeze(), centermap.squeeze()
            
        else:
            masks = np.zeros((confimap.shape[0], confimap.shape[1]))  #pass back zeros if not compute_masks
        
        # print("model_mask:", masks.shape)
        return masks, styles, offsetmap, confimap, centermap

    def loss_fn(self, lbl, y):
        """ loss function between true labels lbl and prediction y
         y[:,:2] : offset amp
         y[:,2] : confidence map
         y[:,3] : center map 
         -------
         lbl[:,0] : binary map
         lbl[:,1] : center map
         lbl[:,2:] : offset map  
        """
        
        binlbl  = self._to_device(lbl[:,0]>.5)
        center = self._to_device(lbl[:,3]) 
        offset = self._to_device(lbl[:,1:3]) 

        loss = self.criterion(y[:,:2], offset) 

        loss2 = self.criterion2(y[:,2], binlbl)

        loss3 = self.criterion(y[:,3], center)

        loss = loss + loss2 + loss3
        return loss

    def train(self, train_data, train_labels, train_files=None, 
              test_data=None, test_labels=None, test_files=None,
              channels=None, normalize=True, 
              save_path=None, save_every=100, save_each=False,
              learning_rate=0.2, n_epochs=500, momentum=0.9, SGD=True,
              weight_decay=0.00001, batch_size=8, nimg_per_epoch=None,
              rescale=True, min_train_masks=5,
              model_name=None):

        train_data, train_labels, test_data, test_labels, run_test = transforms.reshape_train_test(train_data, train_labels,
                                                                                                   test_data, test_labels,
                                                                                                   channels, normalize)
        
        train_flows = dynamics.labels_to_flows(train_labels, files=train_files, use_gpu=self.gpu, device=self.device)
        if run_test:
            test_flows = dynamics.labels_to_flows(test_labels, files=test_files, use_gpu=self.gpu, device=self.device)
        else:
            test_flows = None

        nmasks = np.array([label[0].max() for label in train_flows])
        nmasks_test = np.array([label[0].max() for label in test_flows])

        nremove = (nmasks < min_train_masks).sum()
        if nremove > 0:
            models_logger.warning(f'{nremove} train images with number of masks less than min_train_masks ({min_train_masks}), removing from train set')
            ikeep = np.nonzero(nmasks >= min_train_masks)[0]
            train_data = [train_data[i] for i in ikeep]
            train_flows = [train_flows[i] for i in ikeep]

        nremove_test = (nmasks_test < min_train_masks).sum()
        if nremove_test > 0:
            models_logger.warning(f'{nremove_test} train images with number of masks less than min_train_masks ({min_train_masks}), removing from test set')
            ikeep = np.nonzero(nmasks_test >= min_train_masks)[0]
            test_data = [test_data[i] for i in ikeep]
            test_flows = [test_flows[i] for i in ikeep]

        if channels is None:
            models_logger.warning('channels is set to None, input must therefore have nchan channels (default is 2)')
        model_path = self._train_net(train_data, train_flows, 
                                     test_data=test_data, test_labels=test_flows,
                                     save_path=save_path, save_every=save_every, save_each=save_each,
                                     learning_rate=learning_rate, n_epochs=n_epochs, 
                                     momentum=momentum, weight_decay=weight_decay, 
                                     SGD=SGD, batch_size=batch_size, nimg_per_epoch=nimg_per_epoch, 
                                     rescale=rescale, model_name=model_name)
        self.pretrained_model = model_path
        return model_path
