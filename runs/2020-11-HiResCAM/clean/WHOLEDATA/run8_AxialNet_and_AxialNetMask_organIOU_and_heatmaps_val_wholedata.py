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

import run_attn_analysis
from models import custom_models_base, custom_models_mask
from load_dataset import custom_datasets

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
    #TODO: replace attn_storage_dir with the path to the directory containing the
    #attention ground truth that was saved during the first epoch of training
    attn_storage_dir = '/home/rlb61/data/img-hiermodel2/results/2020-12-12_AxialNetMask_WholeData_dilateFalse_nearest/attn_storage_temp/'
    
    experiments = {
        'AxialNet':{'params_path':'/home/rlb61/data/img-hiermodel2/results/2020-12-12_AxialNet_WholeData/params/AxialNet_WholeData_epoch4', #TODO: replace this with the correct path to the final parameters
                    'stop_epoch':4, #TODO: replace this with the correct int for the final epoch
                    'model_name':'AxialNet',
                    'custom_net':custom_models_base.AxialNet,
                    'target_layer_name':'7'},
        
        'AxialNetMask':{'params_path':'/home/rlb61/data/img-hiermodel2/results/2020-12-12_AxialNetMask_WholeData_dilateFalse_nearest/params/AxialNetMask_WholeData_dilateFalse_nearest_epoch4', #TODO: replace this with the correct path to the final parameters
                       'stop_epoch':4, #TODO: replace this with the correct int for the final epoch
                       'model_name':'AxialNet',
                       'custom_net':custom_models_mask.AxialNet_Mask,
                       'target_layer_name':'7'}}
    
    for key in experiments.keys():
        for task in ['iou_analysis','attn_plots']:
            for attention_type in ['gradcam-vanilla','hirescam']:
                tot0 = timeit.default_timer()
                run_attn_analysis.AttentionAnalysis(task=[task],
                            attention_type=attention_type,
                            attention_type_args={'model_name':experiments[key]['model_name'],
                                                 'target_layer_name':experiments[key]['target_layer_name']},
                            setname='valid',
                            valid_results_dir='', #Do not need a valid_results_dir because this experiment will create a valid_results_dir (specifying the per-label binarization thresholds for OrganIOU)
                            custom_net=experiments[key]['custom_net'],
                            custom_net_args={'n_outputs':80,'slices':135},
                            params_path=experiments[key]['params_path'],
                            stop_epoch=experiments[key]['stop_epoch'],
                            which_scans={},
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
                                'selected_note_acc_files':{'train':'', 'valid':'', 'test':''}})
                tot1 = timeit.default_timer()
                print('Total Time', round((tot1 - tot0)/60.0,2),'minutes')
        