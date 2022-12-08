#2021-03-11-valid-plot-true-positives-with-highest-pred-prob-leftovers-nn.py
#Copyright (c) 2021 Rachel Lea Ballantyne Draelos

#MIT License

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE

import timeit
import pandas as pd
from src import run_attn_analysis
from src.models import custom_models_base, custom_models_mask
from src.load_dataset import custom_datasets
from src.plot import identify_cts_to_plot

def update(which_scans_A, flattened_which_scans_A, flattened_which_scans_B, to_plot_for_B):
    curridx = 0
    for pair in flattened_which_scans_A:
        if pair not in flattened_which_scans_B:
            volumeacc = pair.split('#')[0]
            abnormality = pair.split('#')[1]
            if volumeacc not in to_plot_for_B['VolumeAcc'].values.tolist():
                to_plot_for_B.at[curridx,'VolumeAcc'] = volumeacc
                fake_volacc_list = which_scans_A[which_scans_A['VolumeAcc']==volumeacc]['VolumeAcc_ForOutput'].values.tolist()
                assert len(fake_volacc_list)==1
                to_plot_for_B.at[curridx,'VolumeAcc_ForOutput'] = fake_volacc_list[0]
                to_plot_for_B.at[curridx,'Abnormality'] = abnormality
                curridx+=1
            else:
                chosenidx = to_plot_for_B[to_plot_for_B['VolumeAcc']==volumeacc].index.values.tolist() #e.g. [9]
                assert len(chosenidx)==1
                chosenidx = chosenidx[0] #e.g. 9
                to_plot_for_B.at[chosenidx,'Abnormality'] = to_plot_for_B.at[chosenidx,'Abnormality']+','+abnormality
    return to_plot_for_B

if __name__=='__main__':
    attn_storage_dir = '/home/rlb61/data/img-hiermodel2/results/results_2019-2020/2020-10/2020-10-08_WHOLEDATA_BodyAvg_Mask_dilateFalse_nearest_Lambda33_FreshStart/attn_storage_temp/'
    
    experiments = {
        'AxialNet':{'params_path':'/home/rlb61/data/img-hiermodel2/results/results_2019-2020/2020-10/2020-10-09_WHOLEDATA_BodyAvg_Baseline_FreshStart/params/WHOLEDATA_BodyAvg_Baseline_FreshStart_epoch4',
                    'grtruth_path':'/home/rlb61/data/img-hiermodel2/results/results_2019-2020/2020-10/2020-10-09_WHOLEDATA_BodyAvg_Baseline_FreshStart/pred_probs/valid_grtruth_ep4.csv',
                    'predprob_path':'/home/rlb61/data/img-hiermodel2/results/results_2019-2020/2020-10/2020-10-09_WHOLEDATA_BodyAvg_Baseline_FreshStart/pred_probs/valid_predprob_ep4.csv',
                    'mapping_path':'/home/rlb61/data/img-hiermodel2/data/RADChestCT_Mapping/Conversion_PHI_to_DEID.csv',
                    'results_dir_force':'/home/rlb61/data/img-hiermodel2/results/results_2021/2021-03-10_Valid-Plot-True-Pos-with-Highest-PredProb-2020-10-09-Base-nn/', #NN BECAUSE THAT IS THE CURRENT ALGORITHM IN THE 2D PLOTTING SCRIPT
                    'stop_epoch':4,
                    'model_name':'AxialNet',
                    'custom_net':custom_models_base.AxialNet,
                    'target_layer_name':'7'},
        
        'AxialNetMask':{'params_path':'/home/rlb61/data/img-hiermodel2/results/results_2019-2020/2020-10/2020-10-08_WHOLEDATA_BodyAvg_Mask_dilateFalse_nearest_Lambda33_FreshStart/params/WHOLEDATA_BodyAvg_Mask_dilateFalse_nearest_Lambda33_FreshStart_epoch4',
                        'grtruth_path':'/home/rlb61/data/img-hiermodel2/results/results_2019-2020/2020-10/2020-10-08_WHOLEDATA_BodyAvg_Mask_dilateFalse_nearest_Lambda33_FreshStart/pred_probs/valid_grtruth_ep4.csv',
                        'predprob_path':'/home/rlb61/data/img-hiermodel2/results/results_2019-2020/2020-10/2020-10-08_WHOLEDATA_BodyAvg_Mask_dilateFalse_nearest_Lambda33_FreshStart/pred_probs/valid_predprob_ep4.csv',
                        'mapping_path':'/home/rlb61/data/img-hiermodel2/data/RADChestCT_Mapping/Conversion_PHI_to_DEID.csv',
                        'results_dir_force':'/home/rlb61/data/img-hiermodel2/results/results_2021/2021-03-10_Valid-Plot-True-Pos-with-Highest-PredProb-2020-10-08-Mask-nn/', #NN BECAUSE THAT IS THE CURRENT ALGORITHM IN THE 2D PLOTTING SCRIPT
                        'stop_epoch':4,
                        'model_name':'AxialNet',
                        'custom_net':custom_models_mask.AxialNet_Mask,
                        'target_layer_name':'7'}}
    
    ##########################
    # Figure out what to run #--------------------------------------------------
    ##########################
    #Next time, seriously take the union of these before you do the plotting
    #Now I have to do a complicated "what's left over" in order to prevent
    #having to wait FOREVER for the script to run (i.e. in order to prevent
    #redoing a bunch of work that was already done.)
    which_scans_mask = identify_cts_to_plot.return_true_pos_with_highest_pred_prob(grtruth_path=experiments['AxialNetMask']['grtruth_path'],
                                                                            predprob_path=experiments['AxialNetMask']['predprob_path'],
                                                                            mapping_path=experiments['AxialNetMask']['mapping_path'])
    which_scans_base = identify_cts_to_plot.return_true_pos_with_highest_pred_prob(grtruth_path=experiments['AxialNet']['grtruth_path'],
                                                                            predprob_path=experiments['AxialNet']['predprob_path'],
                                                                            mapping_path=experiments['AxialNet']['mapping_path'])
    
    flattened_which_scans_mask = []
    for idx in which_scans_mask.index.values.tolist():
        volumeacc = which_scans_mask.at[idx,'VolumeAcc']
        abnormalities = which_scans_mask.at[idx,'Abnormality'].split(',')
        flattened_which_scans_mask+=[volumeacc+'#'+abn for abn in abnormalities]
    
    flattened_which_scans_base = []
    for idx in which_scans_base.index.values.tolist():
        volumeacc = which_scans_base.at[idx,'VolumeAcc']
        abnormalities = which_scans_base.at[idx,'Abnormality'].split(',')
        flattened_which_scans_base+=[volumeacc+'#'+abn for abn in abnormalities]
        
    to_plot_for_mask = pd.DataFrame(columns=['VolumeAcc', 'VolumeAcc_ForOutput', 'Abnormality'])
    to_plot_for_base = pd.DataFrame(columns=['VolumeAcc', 'VolumeAcc_ForOutput', 'Abnormality'])
    
    to_plot_for_base = update(which_scans_mask, flattened_which_scans_mask, flattened_which_scans_base, to_plot_for_base)
    to_plot_for_mask = update(which_scans_base, flattened_which_scans_base, flattened_which_scans_mask, to_plot_for_mask)
    
    #######
    # Run #---------------------------------------------------------------------
    #######
    for key in experiments.keys():
        
        if key == 'AxialNetMask':
            chosen_scans = to_plot_for_mask
        elif key == 'AxialNet':
            chosen_scans = to_plot_for_base
        
        for task in ['attn_plots']:
            for attention_type in ['gradcam-vanilla','hirescam']:
                tot0 = timeit.default_timer()
                run_attn_analysis.AttentionAnalysis(
                            results_dir_force = experiments[key]['results_dir_force'],
                            base_results_dir = '/home/rlb61/data/img-hiermodel2/results/results_2021/',
                            task=[task],
                            attention_type=attention_type,
                            attention_type_args={'model_name':experiments[key]['model_name'],
                                                 'target_layer_name':experiments[key]['target_layer_name']},
                            setname='valid',
                            valid_results_dir='', #Do not need a valid_results_dir because this experiment is on the validation set
                            custom_net=experiments[key]['custom_net'],
                            custom_net_args={'n_outputs':80,'slices':135},
                            params_path=experiments[key]['params_path'],
                            stop_epoch=experiments[key]['stop_epoch'],
                            
                            which_scans=chosen_scans,
                            
                            dataset_class=custom_datasets.CTDataset_2019_10,
                            dataset_args={'verbose':False,
                                'label_type_ld':'location_disease_0323',
                                'genericize_lung_labels':True,
                                'label_counts':{'mincount_heart':200,
                                            'mincount_lung':125},
                                'view':'axial',
                                'use_projections9':False,
                                'loss_string':'AxialNet_Mask-loss', #we're not calculating a loss here but we need the masks to be used
                                'volume_prep_args':{
                                            'pixel_bounds':[-1000,800],
                                            'num_channels':3,
                                            'crop_type':'single',
                                            'selfsupervised':False,
                                            'from_seg':False},
                                'attn_gr_truth_prep_args':{
                                            'attn_storage_dir':attn_storage_dir,
                                            'dilate':False,
                                            'downsamp_mode':'nearest'},
                                #Paths
                                #here the which_scans come form the validation set in general, not the subset!
                                'selected_note_acc_files':{'train':'',
                                                           'valid':''},
                                'ct_scan_path':'/storage/rlb61-ct_images/vna/rlb61/2019-10-BigData-numpy/', #DEID version: '/scratch/rlb61/2019-10-BigData-DEID'
                                'ct_scan_projections_path':'/storage/rlb61-ct_images/vna/rlb61/2020-04-15-Projections/', #DEID version: '/scratch/rlb61/2020-04-15-Projections-DEID'
                                'key_csvs_path':'/home/rlb61/data/img-hiermodel2/data/RADChestCT_PHI/', #DEID version: '/home/rlb61/data/img-hiermodel2/data/RADChestCT_DEID/'
                                'segmap_path':'/storage/rlb61-ct_images/vna/rlb61/2019-10-BigData-segmaps-PHI/' #DEID version: '/storage/rlb61-ct_images/vna/rlb61/2019-10-BigData-segmaps-DEID/'
                                })
                tot1 = timeit.default_timer()
                print('Total Time', round((tot1 - tot0)/60.0,2),'minutes')
        