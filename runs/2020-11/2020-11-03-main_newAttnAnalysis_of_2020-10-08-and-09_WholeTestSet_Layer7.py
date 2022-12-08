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
    
    #task can be 'iou_analysis', 'blue_heatmaps', and/or 'attn_plots'
    #attention_type can be 'gradcam-vanilla','gradcam-new-avg', or 'disease-reps-avg'
    experiments = {'10-08-Mask':{'params_path_choice':'img-hiermodel2/results/2020-10/2020-10-08_WHOLEDATA_BodyAvg_Mask_dilateFalse_nearest_Lambda33_FreshStart/params/WHOLEDATA_BodyAvg_Mask_dilateFalse_nearest_Lambda33_FreshStart_epoch4',
                                 'valid_results_dir':'img-hiermodel2/results/2020-11/2020-11-02_ValidAttnAnalysis_of_2020-10-08_WHOLEDATA_BodyAvg_Mask_dilateFalse_nearest_Lambda33_FreshStart'},
                   '10-09-Base':{'params_path_choice':'img-hiermodel2/results/2020-10/2020-10-09_WHOLEDATA_BodyAvg_Baseline_FreshStart/params/WHOLEDATA_BodyAvg_Baseline_FreshStart_epoch4',
                                 'valid_results_dir':'img-hiermodel2/results/2020-11/2020-11-03_ValidAttnAnalysis_of_2020-10-09_WHOLEDATA_BodyAvg_Baseline_FreshStart'}}
    
    for key in experiments.keys():
        print('\n\nRunning',key)
        params_path_choice = experiments[key]['params_path_choice']
        valid_results_dir = experiments[key]['valid_results_dir']
        
        for attention_type in ['gradcam-vanilla','gradcam-new-avg','disease-reps-avg']:
            print('Running',attention_type)
            tot0 = timeit.default_timer()
            run_attn_analysis.AttentionAnalysis(task=['iou_analysis'],
                        attention_type=attention_type,
                        attention_type_args={'model_name':'BodyAvg', #the model outputs class returned for BodyAvg is the same as the model outputs class returned for BodyAvg_Mask so this is fine
                                             'target_layer_name':'7'}, #THIS IS THE ONLY IMPLEMENTATION DIFFERENCE BETWEEN THIS RUN AND THE PREVIOUS TEST SET RUN! USING LAYER 7 INSTEAD OF LAYER 6!!!
                        setname='test', #RUN ON THE TEST SET
                        valid_results_dir=prefix+valid_results_dir,
                        custom_net=custom_models_mask.BodyAvg_Mask,
                        custom_net_args={'n_outputs':80,'slices':135},
                        params_path=prefix+params_path_choice,
                        stop_epoch=4,
                        which_scans=[],#enc_val_scans.return_which_scans(),
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
                            'selected_note_acc_files':{'train':'', 'valid':'', 'test':''}})
            tot1 = timeit.default_timer()
            print('Total Time', round((tot1 - tot0)/60.0,2),'minutes')
    