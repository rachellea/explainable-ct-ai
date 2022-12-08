#run1_AllTable1Models_trainval_smalldata.py
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

from run_experiment import DukeCTExperiment
from models import custom_models_base, custom_models_ablation, custom_models_alternative, custom_models_old
from load_dataset import custom_datasets

#run 1: Table 1 validation set AUROC replication.
#Train the alternative (CTNet, 3DConv, BodyConv), ablated (MaxPool,
#RandInitResNet, NoCustomConv), and proposed (AxialNet) models of
#Table 1, and calculate the validation set AUROC.
#This experiment is performed on the small data subset.

#TODO: change NUM_EPOCHS to 100 to replicate paper results.
NUM_EPOCHS = 1 #temporarily set to 1 for fast code demo

if __name__=='__main__':
    experiments = {
        #Alternative: CTNet, 3DConv, BodyConv
        'CTNet':{'custom_net':custom_models_old.CTNetModel,'num_channels':3},
        '3DConv':{'custom_net':custom_models_alternative.ThreeDConv,'num_channels':1}, #1-channel input because doesn't use pretrained feature extractor
        'BodyConv':{'custom_net':custom_models_alternative.BodyConv,'num_channels':3},
        
        #Ablation: MaxPool, RandInitResNet, NoCustomConv
        'AxialNet_FinalMaxPool':{'custom_net':custom_models_ablation.Ablate_AxialNet_FinalMaxPool, 'num_channels':3},
        'AxialNet_RandomInitResNet':{'custom_net':custom_models_ablation.Ablate_AxialNet_RandomInitResNet,'num_channels':3},
        'AxialNet_NoCustomConv':{'custom_net':custom_models_ablation.Ablate_AxialNet_NoCustomConv,'num_channels':3},
        
        #Proposed: AxialNet
        'AxialNet':{'custom_net':custom_models_base.AxialNet, 'num_channels':3}}
    
    for key in experiments.keys():
        tot0 = timeit.default_timer()
        DukeCTExperiment(descriptor=key+'_SmallData',
            custom_net = experiments[key]['custom_net'],
            custom_net_args = {'n_outputs':80},
            loss_string = 'bce',
            loss_args = {},
            learning_rate = 1e-3,
            weight_decay = 1e-7,
            num_epochs=NUM_EPOCHS, patience = 15,
            batch_size = 1, device = 0,
            data_parallel = False, model_parallel = False,
            use_test_set = False, task = 'train_eval',
            old_params_dir = '',
            dataset_class = custom_datasets.CTDataset_2019_10,
            dataset_args = {
                        'label_type_ld':'location_disease_0323',
                        'genericize_lung_labels':True,
                        'label_counts':{'mincount_heart':200,
                                    'mincount_lung':125},
                        'view':'axial',
                        'use_projections9':False,
                        'volume_prep_args':{
                                    'pixel_bounds':[-1000,800],
                                    'num_channels':experiments[key]['num_channels'],
                                    'crop_type':'single',
                                    'selfsupervised':False,
                                    'from_seg':False},
                        'attn_gr_truth_prep_args':{
                                    'dilate':None,
                                    'downsamp_mode':None},
                        
                        #TODO: replace these with the paths to the files that define
                        #the data subset of 2,000 training and 1,000 validation volumes
                        'selected_note_acc_files':{'train':'/home/rlb61/data/img-hiermodel2/load_dataset/ground_truth/predefined_subsets/2020-01-10-imgtrain_random2000.csv',
                                                   'valid':'/home/rlb61/data/img-hiermodel2/load_dataset/ground_truth/predefined_subsets/2020-01-10-imgvalid_a_random1000.csv'}})
        tot1 = timeit.default_timer()
        print('Total Time', round((tot1 - tot0)/60.0,2),'minutes')
