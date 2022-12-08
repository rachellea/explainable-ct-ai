#main.py

import timeit

import enc_val_scans
import attention_iou_analysis
from models import custom_models_mask
from load_dataset import custom_datasets

if __name__=='__main__':
    lambda_val = 1.0/3.0
    downsamp_mode = 'nearest'
    dilate = False
    
    tot0 = timeit.default_timer()
    attention_iou_analysis.CalculateIOUAndHeatmapsFinalEpoch(
        custom_net = custom_models_mask.BodyAvg_Mask,
        custom_net_args = {'n_outputs':80,'slices':15},
        params_path = '/home/rlb61/data/img-hiermodel2/results/2020-09/2020-09-18_BodyAvg_Updates/params/BodyAvg_Updates',
        stop_epoch = 'Final',
        which_scans = enc_val_scans.return_which_scans(),
        dataset_class = custom_datasets.CTDataset_2019_10,
        dataset_args = {'verbose':False,
                    'label_type_ld':'location_disease_0323',
                    'genericize_lung_labels':True,
                    'label_counts':{'mincount_heart':200, #default 200
                                'mincount_lung':125}, #default 125
                    'view':'axial',
                    'use_projections9':True,
                    'loss_string':'BodyAvg_Mask-loss', #we're not calculating a loss here but we need the masks to be used
                    'volume_prep_args':{
                                'pixel_bounds':[-1000,800], #Recent increase in upper pixel bound to 800 -> better perf
                                'num_channels':3,
                                'crop_type':'single',
                                'selfsupervised':False,
                                'from_seg':False},
                    'attn_gr_truth_prep_args':{
                                'attn_storage_dir':'/home/rlb61/data/img-hiermodel2/results/2020-09/2020-09-27_BodyAvg_Mask_CORRECT_dilateFalse_nearest/attn_storage_temp/',
                                'dilate':dilate,
                                'downsamp_mode':downsamp_mode},
                    'selected_note_acc_files':{'train':'/home/rlb61/data/img-hiermodel2/load_dataset/ground_truth/predefined_subsets/2020-01-10-imgtrain_random2000.csv',
                                                       'valid':'/home/rlb61/data/img-hiermodel2/load_dataset/ground_truth/predefined_subsets/2020-01-10-imgvalid_a_random1000.csv'}})
    tot1 = timeit.default_timer()
    print('Total Time', round((tot1 - tot0)/60.0,2),'minutes')
    