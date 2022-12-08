#2021-03-13-run3_BodyCAM_organIOU_val_smalldata.py
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

import pandas as pd

from src import run_attn_analysis
from src.models import custom_models_base
from src.load_dataset import custom_datasets

if __name__=='__main__':
    attn_storage_dir = '/home/rlb61/data/img-hiermodel2/results/results_2019-2020/2020-10/2020-10-08_WHOLEDATA_BodyAvg_Mask_dilateFalse_nearest_Lambda33_FreshStart/attn_storage_temp/'
    
    experiments = {
        'BodyCAM':{'params_path':'/home/rlb61/data/img-hiermodel2/results/results_2019-2020/2020-11/2020-11-10_OTHER_BodyCAM/params/OTHER_BodyCAM_epoch30',
                       'stop_epoch':30,
                       'model_name':'BodyCAM',
                       'custom_net':custom_models_base.BodyCAM,
                       'target_layer_name':'7', #7 is the last layer of convolution
                       'num_channels':3},
        }
    
    for key in experiments.keys():
        for attention_type in ['gradcam-vanilla','hirescam']:
            tot0 = timeit.default_timer()
            run_attn_analysis.AttentionAnalysis(
                        results_dir_force = '/home/rlb61/data/img-hiermodel2/results/results_2021/2021-03-13_ValidAttnAnalysis_of_2020-11-10_OTHER_BodyCAM/',
                        base_results_dir = '/home/rlb61/data/img-hiermodel2/results/results_2021/',
                        task=['iou_analysis'],
                        attention_type=attention_type,
                        attention_type_args={'model_name':experiments[key]['model_name'],
                                             'target_layer_name':experiments[key]['target_layer_name']},
                        setname='valid',
                        valid_results_dir='',
                        custom_net=experiments[key]['custom_net'],
                        custom_net_args={'n_outputs':80,'slices':135},
                        params_path=experiments[key]['params_path'],
                        stop_epoch=experiments[key]['stop_epoch'],
                        which_scans=pd.DataFrame(),
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
                                        'num_channels':experiments[key]['num_channels'],
                                        'crop_type':'single',
                                        'selfsupervised':False,
                                        'from_seg':False},
                            'attn_gr_truth_prep_args':{
                                        'attn_storage_dir':attn_storage_dir,
                                        'dilate':False,
                                        'downsamp_mode':'nearest',
                                        'small_square':6},
                            
                            #Paths
                            'selected_note_acc_files':{'train':'/home/rlb61/data/img-hiermodel2/data/RADChestCT_PHI/predefined_subsets/2020-01-10-imgtrain_random2000.csv',
                                               'valid':'/home/rlb61/data/img-hiermodel2/data/RADChestCT_PHI/predefined_subsets/2020-01-10-imgvalid_a_random1000.csv'},
                            'ct_scan_path':'/storage/rlb61-ct_images/vna/rlb61/2019-10-BigData-numpy/', #DEID version: '/scratch/rlb61/2019-10-BigData-DEID'
                            'ct_scan_projections_path':'/storage/rlb61-ct_images/vna/rlb61/2020-04-15-Projections/', #DEID version: '/scratch/rlb61/2020-04-15-Projections-DEID'
                            'key_csvs_path':'/home/rlb61/data/img-hiermodel2/data/RADChestCT_PHI/', #DEID version: '/home/rlb61/data/img-hiermodel2/data/RADChestCT_DEID/'
                            'segmap_path':'/storage/rlb61-ct_images/vna/rlb61/2019-10-BigData-segmaps-PHI/' #DEID version: '/storage/rlb61-ct_images/vna/rlb61/2019-10-BigData-segmaps-DEID/'
                            })
            tot1 = timeit.default_timer()
            print('Total Time', round((tot1 - tot0)/60.0,2),'minutes')
    