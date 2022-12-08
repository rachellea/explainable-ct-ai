#main.py

import timeit

import enc_val_scans
import run_attn_analysis
from models import custom_models_old
from load_dataset import custom_datasets

if __name__=='__main__':
    machine = 19
    if machine == 9:
        prefix = '/home/rlb61/data/'
    elif machine == 19:
        prefix = '/storage/rlb61-data/'
    
    #task can be 'iou_analysis', 'blue_heatmaps', and/or 'attn_plots'
    #attention_type can be 'gradcam-vanilla','gradcam-new-avg', or 'disease-reps-avg'
    for attention_type in ['gradcam-vanilla','gradcam-new-avg']:
        tot0 = timeit.default_timer()
        run_attn_analysis.AttentionAnalysis(task=['iou_analysis'],
                    attention_type=attention_type,
                    attention_type_args={'model_name':'ResNet18_Original',
                                         'target_layer_name':'7'},
                    setname='valid',
                    valid_results_dir='',
                    custom_net=custom_models_old.ResNet18_Original,
                    custom_net_args={'n_outputs':80,'slices':135},
                    params_path=prefix+'img-hiermodel2/results/2020-10-24_WHOLEDATA_OldCTNet_Retry/params/WHOLEDATA_OldCTNet_Retry_epoch20',
                    stop_epoch=20,
                    which_scans= [],
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
        
    
    
    
    