#run3_AllTable1Models_organIOU_val_smalldata.py
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
from models import custom_models_base, custom_models_mask, custom_models_ablation, custom_models_alternative, custom_models_old
from load_dataset import custom_datasets

#run3: Table 1 validation set OrganIOU replication.
#Use the saved parameters of the trained models from run1_AllTable1Models_trainval_smalldata.py
#and the saved attention ground truth from run2_TableS3_AxialNetMaskVariants_trainval_smalldata.py
#(with dilate=False and downsamp_mode='nearest') in order to calculate
#organIOU for all of the Table 1 models.
#This experiment is performed on the small data subset.

if __name__=='__main__':
    #TODO: replace attn_storage_dir with the path to the directory containing the
    #final attention ground truth that was saved during run2_TableS3_AxialNetMaskVariants_trainval_smalldata.py
    #The settings dilate=False and downsamp_mode='nearest' were used for the
    #final attention ground truth
    attn_storage_dir = '/home/rlb61/data/img-hiermodel2/results/2020-12-12_AxialNetMask_SmallData_dilateFalse_nearest/attn_storage_temp/'
    
    experiments = {
        #Alternative: CTNet, 3DConv, BodyConv
        #Cannot calculate organIOU for last conv layer of CTNet because
        #spatial relationship with input has been destroyed.
        '3DConv':{'params_path':'/home/rlb61/data/img-hiermodel2/results/2020-12-12_3DConv_SmallData/params/3DConv_SmallData_epoch14', #TODO: replace this with the correct path to the final parameters
                       'stop_epoch':14, #TODO: replace this with the correct int for the final epoch
                       'model_name':'ThreeDConv',
                       'custom_net':custom_models_alternative.ThreeDConv,
                       'target_layer_name':'11', #11 is the last convolutional layer of this ThreeDConv model
                       'num_channels':1}, #1-channel input because doesn't use pretrained feature extractor
        
        'BodyConv':{'params_path':'/home/rlb61/data/img-hiermodel2/results/2020-12-12_BodyConv_SmallData/params/BodyConv_SmallData_epoch6', #TODO: replace this with the correct path to the final parameters
                       'stop_epoch':6, #TODO: replace this with the correct int for the final epoch
                       'model_name':'BodyConv',
                       'custom_net':custom_models_alternative.BodyConv,
                       'target_layer_name':'5', #5 is the last layer of the reducingconvs
                       'num_channels':3},
        
        #Ablation: MaxPool, RandInitResNet, NoCustomConv
        'AxialNet_FinalMaxPool':{'params_path':'/home/rlb61/data/img-hiermodel2/results/2020-12-12_AxialNet_FinalMaxPool_SmallData/params/AxialNet_FinalMaxPool_SmallData_epoch36', #TODO: replace this with the correct path to the final parameters
                       'stop_epoch':36, #TODO: replace this with the correct int for the final epoch
                       'model_name':'AxialNet',
                       'custom_net':custom_models_base.AxialNet, #use AxialNet here to avoid issue of only one max slice per disease being chosen
                       'target_layer_name':'7',
                       'num_channels':3},
        
        'AxialNet_RandomInitResNet':{'params_path':'/home/rlb61/data/img-hiermodel2/results/2020-12-12_AxialNet_RandomInitResNet_SmallData/params/AxialNet_RandomInitResNet_SmallData_epoch54', #TODO: replace this with the correct path to the final parameters
                       'stop_epoch':54, #TODO: replace this with the correct int for the final epoch
                       'model_name':'AxialNet',
                       'custom_net':custom_models_ablation.Ablate_AxialNet_RandomInitResNet,
                       'target_layer_name':'7',
                       'num_channels':3},
        
        'AxialNet_NoCustomConv':{'params_path':'/home/rlb61/data/img-hiermodel2/results/2020-12-12_AxialNet_NoCustomConv_SmallData/params/AxialNet_NoCustomConv_SmallData_epoch19', #TODO: replace this with the correct path to the final parameters
                       'stop_epoch':19, #TODO: replace this with the correct int for the final epoch
                       'model_name':'NoCustomConv',
                       'custom_net':custom_models_ablation.Ablate_AxialNet_NoCustomConv,
                       'target_layer_name':'7', #it actually is also layer 7 for the last conv layer of the ResNet-18 of NoCustomConv
                       'num_channels':3},
        
        #Proposed: AxialNet
        'AxialNet':{'params_path':'/home/rlb61/data/img-hiermodel2/results/2020-12-12_AxialNet_SmallData/params/AxialNet_SmallData_epoch36', #TODO: replace this with the correct path to the final parameters
                       'stop_epoch':36, #TODO: replace this with the correct int for the final epoch
                       'model_name':'AxialNet',
                       'custom_net':custom_models_base.AxialNet,
                       'target_layer_name':'7',
                       'num_channels':3}}
    
    for key in experiments.keys():
        for attention_type in ['gradcam-vanilla','hirescam']:
            tot0 = timeit.default_timer()
            run_attn_analysis.AttentionAnalysis(task=['iou_analysis'],
                        attention_type=attention_type,
                        attention_type_args={'model_name':experiments[key]['model_name'],
                                             'target_layer_name':experiments[key]['target_layer_name']},
                        setname='valid',
                        valid_results_dir='',
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
                                        'num_channels':experiments[key]['num_channels'],
                                        'crop_type':'single',
                                        'selfsupervised':False,
                                        'from_seg':False},
                            'attn_gr_truth_prep_args':{
                                        'attn_storage_dir':attn_storage_dir,
                                        'dilate':False,
                                        'downsamp_mode':'nearest'},
                            
                            #TODO: replace these with the paths to the files that define
                            #the data subset of 2,000 training and 1,000 validation volumes
                            'selected_note_acc_files':{'train':'/home/rlb61/data/img-hiermodel2/load_dataset/ground_truth/predefined_subsets/2020-01-10-imgtrain_random2000.csv',
                                                       'valid':'/home/rlb61/data/img-hiermodel2/load_dataset/ground_truth/predefined_subsets/2020-01-10-imgvalid_a_random1000.csv'}})
            tot1 = timeit.default_timer()
            print('Total Time', round((tot1 - tot0)/60.0,2),'minutes')
    