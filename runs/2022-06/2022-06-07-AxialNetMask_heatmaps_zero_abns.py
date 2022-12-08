#2022-06-07-AxialNetMask_heatmaps_zero_abns.py
#Copyright (c) 2022 Rachel Lea Ballantyne Draelos

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

import os
import timeit
import pandas as pd

from src import run_attn_analysis
from src.models import custom_models_base, custom_models_mask
from src.load_dataset import custom_datasets

import enc_scan_names

#this is a configuration to make plots for the AxialNet model that doesn't use
#the mask loss, in case I want to make plots from that model in the future.
#For now, I'm only going to make plots for the AxialNet model WITH the mask
#loss i.e. not this model:
not_used_experiment = {'AxialNet':{
                    'params_path':'/home/rlb61/data/img-hiermodel2/results/results_2019-2020/2020-10-WHOLEDATA/2020-10-09_WHOLEDATA_BodyAvg_Baseline_FreshStart_TEST/params/WHOLEDATA_BodyAvg_Baseline_FreshStart_epoch4',
                    'stop_epoch':4,
                    'model_name':'AxialNet',
                    'custom_net':custom_models_base.AxialNet,
                    'target_layer_name':'7',
                    #I made a copy of the original valid results dir and appended -COPY to it so that I don't accidentally overwrite anything in the original dir
                    #Although, I think that stuff should only be READ out of valid_results_dir for this run,
                    #and not written...
                    'valid_results_dir':'/home/rlb61/data/img-hiermodel2/results/results_2019-2020/2020-11/2020-11-03_ValidAttnAnalysis_of_2020-10-09_WHOLEDATA_BodyAvg_Baseline_FreshStart_LAYER7-COPY'
                    }}

#note about the params_path:
#the params_path needs to be a copy of the params stored in the directory in which
#the test set abnormality predictions are stored (the _TEST directory for
#these models). why? Because annoyingly in my code I set it up so
#that the directory of the abnormality predictions is inferred
#from the params_path. So if you give the script a params_path
#that corresponds to a validation set run, then it can't find
#test set predictions and it breaks. ugh! I copied the params
#into the test set dir so that this will work.

if __name__=='__main__':
    #attn_storage_dir has to be from the time that I ran stuff on the test set
    #because these are test set volumes for which I want to make visualizations.
    #The visualizations don't depend on the attn gr truth but the code that
    #makes the visualizations does read in the attn gr truth for other reasons
    #which is why I need to provide an attn_storage_dir that already contains
    #the attn gr truth for the appropriate set (train, valid, or test)
    #of the volumes I'm specifying.
    attn_storage_dir = '/home/rlb61/data/img-hiermodel2/results/results_2019-2020/2020-10-WHOLEDATA/2020-10-08_WHOLEDATA_BodyAvg_Mask_dilateFalse_nearest_Lambda33_FreshStart_TEST/attn_storage_temp/'
    
    experiment = {
        'AxialNetMask':{'params_path':'/home/rlb61/data/img-hiermodel2/results/results_2019-2020/2020-10-WHOLEDATA/2020-10-08_WHOLEDATA_BodyAvg_Mask_dilateFalse_nearest_Lambda33_FreshStart_TEST/params/WHOLEDATA_BodyAvg_Mask_dilateFalse_nearest_Lambda33_FreshStart_epoch4',
                       'stop_epoch':4,
                       'model_name':'AxialNet',
                       'custom_net':custom_models_mask.AxialNet_Mask,
                       'target_layer_name':'7',
                       'valid_results_dir':'/home/rlb61/data/img-hiermodel2/results/results_2019-2020/2020-11/2020-11-02_ValidAttnAnalysis_of_2020-10-08_WHOLEDATA_BodyAvg_Mask_dilateFalse_nearest_Lambda33_FreshStart_LAYER7-COPY'
                       }}
    
    for key in experiment.keys():
        
        results_dir_force = '/home/rlb61/data/img-hiermodel2/results/results_2022/2022-06-07-'+key+'-Visualize-Zero-Abn-CT-Scans-Test-Set'
        if not os.path.exists(results_dir_force):
            os.mkdir(results_dir_force)
        
        for attention_type in ['gradcam-vanilla','hirescam']:
            tot0 = timeit.default_timer()
            run_attn_analysis.AttentionAnalysis(
                        results_dir_force=results_dir_force,
                        base_results_dir='', #ignored
                        task=['attn_plots'],
                        attention_type=attention_type,
                        attention_type_args={'model_name':experiment[key]['model_name'],
                                             'target_layer_name':experiment[key]['target_layer_name']},
                        setname='test',
                        valid_results_dir=experiment[key]['valid_results_dir'], #I do need a valid_results_dir because I'm running something on the test set and that depends on thresholds calculated on the validation set
                        custom_net=experiment[key]['custom_net'],
                        custom_net_args={'n_outputs':80,'slices':135},
                        params_path=experiment[key]['params_path'],
                        stop_epoch=experiment[key]['stop_epoch'],
                        
                        which_scans=enc_scan_names.return_scan_names(),
                        
                        dataset_class=custom_datasets.CTDataset_2019_10,
                        dataset_args={'verbose':False,
                                'label_type_ld':'location_disease_0323', #correct
                                'genericize_lung_labels':True, #correct
                                'label_counts':{'mincount_heart':200, #correct
                                            'mincount_lung':125}, #correct
                                'view':'axial', #correct
                                'crop_magnitude':'original', #correct
                                'use_projections9':False, #correct
                                'loss_string':'AxialNet_Mask-loss', #we're not calculating a loss here but loss_string must have 'mask' in it to calculate attn gr truth
                                'volume_prep_args':{
                                            'pixel_bounds':[-1000,800], #correct
                                            'num_channels':3, #correct
                                            'crop_type':'single', #correct
                                            'selfsupervised':False, #correct
                                            'from_seg':False}, #correct
                                'attn_gr_truth_prep_args':{
                                        'attn_storage_dir':attn_storage_dir,
                                        'dilate':False, #correct
                                        'downsamp_mode':'nearest'}, #correct
                                #Paths
                                'selected_note_acc_files':{'train':'',
                                                           'valid':'',
                                                           'test':''},
                                'ct_scan_path':'/storage/rlb61-ct_images/vna/rlb61/2019-10-BigData-numpy',
                                'ct_scan_projections_path':'/storage/rlb61-ct_images/vna/rlb61/2020-04-15-Projections',
                                'key_csvs_path':'/home/rlb61/data/img-hiermodel2/data/RADChestCT_PHI/',
                                'segmap_path':'/storage/rlb61-ct_images/vna/rlb61/2019-10-BigData-segmaps-PHI/'}
                            )
            tot1 = timeit.default_timer()
            print('Total Time', round((tot1 - tot0)/60.0,2),'minutes')
            
