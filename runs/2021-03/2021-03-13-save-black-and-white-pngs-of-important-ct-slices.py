#2021-03-13-save-black-and-white-pngs-of-important-ct-slices.py

import os
from src.plot import visualize_volumes

if __name__=='__main__':
    ct_scan_path = '/scratch/rlb61/2019-10-BigData-DEID'
    results_dir = '/home/rlb61/data/img-hiermodel2/results/results_2021/2021-03-13-Black-and-White-PNGs-of-Important-CT-Slices'
    
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    
    #these filenames are the names of the CT scans currently used in figures in
    #the paper
    figure_1_filenames = ['val27137.npz', #lung pleural effusion
        'val26090.npz'] #234_lung_atelectasis.png
    
    results_figure_filenames = ['val26117.npz', #val26117_lung_groundglass.png
                                'val26876.npz', #val26876_lung_pulmonary_edema.png
                                'val27352.npz', #val27352_lung_aspiration.png
                                'val26395.npz', #val26395_lung_opacity.png
                                'val26235.npz', #chEEE_lung_interstitial_lung_disease.png
                                'val26828.npz', #chFFF_lung_interstitial_lung_disease.png
                                'val26097.npz', #val26097_lung_nodule.png
                                'val26836.npz'] #val26836_heart_catheter_or_port.png
    
    for filename in figure_1_filenames+results_figure_filenames:
        visualize_volumes.save_pngs_of_all_slices(ct_scan_path, filename, results_dir)