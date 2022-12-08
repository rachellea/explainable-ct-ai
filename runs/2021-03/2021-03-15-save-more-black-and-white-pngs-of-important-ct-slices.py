#2021-03-13-save-black-and-white-pngs-of-important-ct-slices.py

import os
from src.plot import visualize_volumes

if __name__=='__main__':
    ct_scan_path = '/scratch/rlb61/2019-10-BigData-DEID'
    results_dir = '/home/rlb61/data/img-hiermodel2/results/results_2021/2021-03-13-Black-and-White-PNGs-of-Important-CT-Slices'
    
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    
    more_filenames = ['val27248.npz', #pericardial thickening
                      'val25926.npz', #pleural thickening
                      'val27086.npz', #lung lucency
                      'val26996.npz'] #lung nodule

    for filename in more_filenames:
        visualize_volumes.save_pngs_of_all_slices(ct_scan_path, filename, results_dir)