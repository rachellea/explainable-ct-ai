#run8_AxialNet_and_AxialNetMask_organIOU_and_heatmaps_val_wholedata.py
#Copyright (c) 2020 Rachel Lea Ballantyne Draelos

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
import enc_all_chosen_scans_PHI_remainder

from src import run_attn_analysis
from src.models import custom_models_base, custom_models_mask
from src.load_dataset import custom_datasets

#run8: Calculate the validation set OrganIOU for the full data AxialNet
#and AxialNet mask loss models. This step is a prerequisite for ultimately
#calculating the Table 2 test set OrganIOU. We need to calculate
#the validation set OrganIOU first because we need to calculate the optimal
#attention map binarization threshold for each label on the validation set
#and then use these optimal thresholds in the test set calculation.
#Use the saved parameters of the trained models from run4_AxialNet_trainval_wholedata.py
#and run5_AxialNetMask_trainval_wholedata.py and the saved attention ground truth
#from run5_AxialNetMask_trainval_wholedata.py to calculate the validation set
#OrganIOU for the AxialNet and AxialNet mask loss models.
#Also make heatmap visualizations of the attention on the validation set. Some
#of these heatmaps are shown in Figure 1 and Figure 5.

if __name__=='__main__':
    attn_storage_dir = '/home/rlb61/data/img-hiermodel2/results/results_2019-2020/2020-10/2020-10-08_WHOLEDATA_BodyAvg_Mask_dilateFalse_nearest_Lambda33_FreshStart/attn_storage_temp/'
    
    experiments = {
        'AxialNet':{'params_path':'/home/rlb61/data/img-hiermodel2/results/results_2019-2020/2020-10/2020-10-09_WHOLEDATA_BodyAvg_Baseline_FreshStart/params/WHOLEDATA_BodyAvg_Baseline_FreshStart_epoch4',
                    'results_dir_force':'/home/rlb61/data/img-hiermodel2/results/results_2021/2021-01-23_ValidAttnAnalysis_of_2020-10-09_WHOLEDATA_BodyAvg_Baseline_FreshStart/',
                    'stop_epoch':4,
                    'model_name':'AxialNet',
                    'custom_net':custom_models_base.AxialNet,
                    'target_layer_name':'7'},
        
        'AxialNetMask':{'params_path':'/home/rlb61/data/img-hiermodel2/results/results_2019-2020/2020-10/2020-10-08_WHOLEDATA_BodyAvg_Mask_dilateFalse_nearest_Lambda33_FreshStart/params/WHOLEDATA_BodyAvg_Mask_dilateFalse_nearest_Lambda33_FreshStart_epoch4',
                     'results_dir_force':'/home/rlb61/data/img-hiermodel2/results/results_2021/2021-01-23_ValidAttnAnalysis_of_2020-10-08_WHOLEDATA_BodyAvg_Mask_dilateFalse_nearest_Lambda33_FreshStart/',
                     'stop_epoch':4,
                     'model_name':'AxialNet',
                     'custom_net':custom_models_mask.AxialNet_Mask,
                     'target_layer_name':'7'}}
    
    for key in experiments.keys():
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
                            valid_results_dir='', #Do not need a valid_results_dir because this experiment will create a valid_results_dir (specifying the per-label binarization thresholds for OrganIOU)
                            custom_net=experiments[key]['custom_net'],
                            custom_net_args={'n_outputs':80,'slices':135},
                            params_path=experiments[key]['params_path'],
                            stop_epoch=experiments[key]['stop_epoch'],
                            which_scans=enc_all_chosen_scans_PHI_remainder.return_which_scans(),
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
        