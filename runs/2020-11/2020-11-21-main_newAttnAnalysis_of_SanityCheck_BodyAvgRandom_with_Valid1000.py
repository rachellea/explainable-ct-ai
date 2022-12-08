#main.py

import timeit

import enc_val_scans
import run_attn_analysis
from models import custom_models_mask
from load_dataset import custom_datasets

if __name__=='__main__':
    machine = 19
    if machine == 9:
        prefix = '/home/rlb61/data/'
    elif machine == 19:
        prefix = '/storage/rlb61-data/'
            
    for attention_type in ['gradcam-vanilla','gradcam-new-avg']:
        print('WORKING ON EXPERIMENT', attention_type)
        tot0 = timeit.default_timer()
        run_attn_analysis.AttentionAnalysis(task=['iou_analysis'],
                    attention_type=attention_type,
                    attention_type_args={'model_name':'BodyAvg', #the model outputs class returned for BodyAvg is the same as the model outputs class returned for BodyAvg_Mask so this is fine
                                         'target_layer_name':'7'}, #THIS IS THE ONLY IMPLEMENTATION DIFFERENCE BETWEEN THIS RUN AND THE PREVIOUS VALIDATION SET RUN! USING LAYER 7 INSTEAD OF LAYER 6!!!
                    setname='valid',
                    valid_results_dir='',
                    custom_net=custom_models_mask.BodyAvg_Mask,
                    custom_net_args={'n_outputs':80,'slices':135},
                    params_path=prefix+'img-hiermodel2/results/2020-11-21-Random-Parameters-BodyAvg/bodyavg_randparams_ckpt',
                    stop_epoch=0,
                    which_scans={},#enc_val_scans.return_which_scans(),
                    dataset_class=custom_datasets.CTDataset_2019_10,
                    dataset_args={'verbose':False,
                        'label_type_ld':'location_disease_0323',
                        'genericize_lung_labels':True,
                        'label_counts':{'mincount_heart':200, #default 200
                                    'mincount_lung':125}, #default 125
                        'view':'axial',
                        'use_projections9':False,
                        'loss_string':'BodyAvg_Mask-loss', #we're not calculating a loss here but we need the masks to be used
                        'volume_prep_args':{
                                    'pixel_bounds':[-1000,800],
                                    'num_channels':3,
                                    'crop_type':'single',
                                    'selfsupervised':False,
                                    'from_seg':False},
                        'attn_gr_truth_prep_args':{
                                    'attn_storage_dir':prefix+'img-hiermodel2/results/2020-10/2020-10-08_WHOLEDATA_BodyAvg_Mask_dilateFalse_nearest_Lambda33_FreshStart/attn_storage_temp/',
                                    'dilate':False,
                                    'downsamp_mode':'nearest'},
                        'selected_note_acc_files':{'train':'/storage/rlb61-data/img-hiermodel2/load_dataset/ground_truth/predefined_subsets/2020-01-10-imgtrain_random2000.csv',
                                           'valid':'/storage/rlb61-data/img-hiermodel2/load_dataset/ground_truth/predefined_subsets/2020-01-10-imgvalid_a_random1000.csv'}})
        tot1 = timeit.default_timer()
        print('Total Time', round((tot1 - tot0)/60.0,2),'minutes')
