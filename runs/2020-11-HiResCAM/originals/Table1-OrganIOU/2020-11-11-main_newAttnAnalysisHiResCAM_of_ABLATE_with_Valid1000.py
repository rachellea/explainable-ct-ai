#main.py

import timeit

import enc_val_scans
import run_attn_analysis
from models import custom_models_mask, custom_models_ablation
from load_dataset import custom_datasets

if __name__=='__main__':
    machine = 19
    if machine == 9:
        prefix = '/home/rlb61/data/'
    elif machine == 19:
        prefix = '/storage/rlb61-data/'
    
    #task can be 'iou_analysis', 'blue_heatmaps', and/or 'attn_plots'
    #attention_type can be 'gradcam-vanilla','gradcam-new-avg', or 'disease-reps-avg'
    experiments = {
        #2020-10-29_ABLATE_BodyAvg_SigmoidAfterCustomConv_Full405_epoch0
        'ABLSigmoidAfterConv':{'params_path_choice':'img-hiermodel2/results/2020-10-ABLATE/2020-10-29_ABLATE_BodyAvg_SigmoidAfterCustomConv_Full405/params/ABLATE_BodyAvg_SigmoidAfterCustomConv_Full405_epoch0',
                               'custom_net':custom_models_mask.BodyAvg_Mask,
                               'model_name':'BodyAvg',
                               'stop_epoch':0},
        
        #2020-10-30_ABLATE_BodyAvg_RandomInitResNet_Full405_epoch54
        'ABLRandInitResNet':{'params_path_choice':'img-hiermodel2/results/2020-10-ABLATE/2020-10-30_ABLATE_BodyAvg_RandomInitResNet_Full405/params/ABLATE_BodyAvg_RandomInitResNet_Full405_epoch54',
                             'custom_net':custom_models_mask.BodyAvg_Mask,
                             'model_name':'BodyAvg',
                             'stop_epoch':54},
        
        #2020-11-02_ABLATE_BodyAvg_FinalMaxPool_Full405_epoch36
        'ABLMaxPool':{'params_path_choice':'img-hiermodel2/results/2020-10-ABLATE/2020-11-02_ABLATE_BodyAvg_FinalMaxPool_Full405/params/ABLATE_BodyAvg_FinalMaxPool_Full405_epoch36',
                      'custom_net':custom_models_mask.BodyAvg_Mask,
                      'model_name':'BodyAvg',
                      'stop_epoch':36},
        
          #2020-11-04_ABLATE_BodyAvg_NoCustomConv_Full405_epoch19
        'ABLNoCustomConv':{'params_path_choice':'img-hiermodel2/results/2020-10-ABLATE/2020-11-04_ABLATE_BodyAvg_NoCustomConv_Full405/params/ABLATE_BodyAvg_NoCustomConv_Full405_epoch19',
                           'custom_net':custom_models_ablation.Ablate_BodyAvg_NoCustomConv,
                           'model_name':'NoCustomConv',
                           'stop_epoch':19}} 
    
    for key in experiments.keys():                
        for attention_type in ['gradcam-new-avg']: #HiResCAM
            print('WORKING ON EXPERIMENT',key, '\n\t', experiments[key]['params_path_choice'], '\n\t', attention_type)
            
            tot0 = timeit.default_timer()
            run_attn_analysis.AttentionAnalysis(task=['iou_analysis'],
                        attention_type=attention_type,
                        attention_type_args={'model_name':experiments[key]['model_name'],
                                             'target_layer_name':'7'}, #it actually is also layer 7 for the last conv layer of the ResNet-18 of NoCustomConv
                        setname='valid',
                        valid_results_dir='',
                        custom_net=experiments[key]['custom_net'],
                        custom_net_args={'n_outputs':80,'slices':135},
                        params_path=prefix+experiments[key]['params_path_choice'],
                        stop_epoch=experiments[key]['stop_epoch'],
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
    