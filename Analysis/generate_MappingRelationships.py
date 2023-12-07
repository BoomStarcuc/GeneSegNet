import os
import cv2
import random
import numpy as np
import glob
import matplotlib.pyplot as plt
import h5py
import csv
import fastremap
import scipy.io as scio
from natsort import natsorted

def main(spot_dir, label_dir, save_dir):
    label_files = natsorted(glob.glob(os.path.join(label_dir, '*.png'))) # .png or .mat
    gene_spot_files = natsorted(glob.glob(os.path.join(spot_dir, '*.csv')))

    assert len(label_files) == len(gene_spot_files)

    print("label_files:", len(label_files))
    print("gene_spot_files:", len(gene_spot_files))
    
    for index, (label_file, gene_spot_file) in enumerate(zip(label_files, gene_spot_files)):
        print("label_file:",label_file)
        print("gene_spot_file:",gene_spot_file)

        image_id = os.path.splitext(os.path.basename(gene_spot_file))[0]
        print("image_id:", image_id)
        
        
        #if you want to use predicted .mat file, please uncomment the following code
        # label = scio.loadmat(label_file)
        # label = np.array(label['CellMap'])

        #if you want to use predicted .png file, please uncomment the following code
        label = cv2.imread(label_file, -1)


        with open(gene_spot_file, 'r') as f:
            lines = f.readlines()
            lines = lines[1:]

        spot_list = []
        for line in lines:
            splits = line.strip().split(',')
            spot_list.append([splits[0], splits[1], splits[2]])
        
        spot_list = np.array(spot_list)
        cell_ids = np.unique(label)

        xs = spot_list[:, 0].astype(np.float64).astype(np.int32)
        ys = spot_list[:, 1].astype(np.float64).astype(np.int32)
        genes = spot_list[:, 2]

        RNA_list = []
        for cell_id in cell_ids:
            gene_mask = label[ys, xs] == cell_id
            cell_genes = spot_list[gene_mask]

            ys_1 = ys[gene_mask]
            xs_1 = xs[gene_mask]

            cell_id_list = np.ones((cell_genes.shape[0],1))*cell_id

            RNA = np.column_stack((cell_id_list, cell_genes))
            RNA_list.append(RNA)

        RNA_list = np.concatenate(np.array(RNA_list))
        print("RNA_list:", RNA_list.shape)
        
        with open( '{}/{}.csv'.format(save_dir, image_id), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["cell_id", 'spotX', 'spotY', 'gene'])
            writer.writerows(RNA_list)
        

if __name__ == '__main__':
    spot_dir = 'D:/medicalproject/code/rebuttal_submit_code/preprocessing_testdata/spots' #the path of raw spot data
    label_dir = 'D:/medicalproject/code/rebuttal_submit_code/inference_results' #the path of generated inference
    save_dir = 'D:/medicalproject/code/rebuttal_submit_code/mappingrelationship'
    os.makedirs(save_dir, exist_ok=True)
    main(spot_dir, label_dir, save_dir)
