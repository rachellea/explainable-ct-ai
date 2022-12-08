#run2_TableS3_AxialNetMaskVariants_trainval_smalldata.py
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

from models import custom_models_mask
from load_dataset import custom_datasets
from run_experiment import DukeCTExperiment

#run2: Table S3 validation set AUROC replication.
#Train the different AxialNet mask loss model variants. The variants use
#different downsampling algorithms to downsample the organ masks
#(area vs. trilinear vs. nearest neighbors algorithms) and different
#use of morphological dilation on the downsampled mask (dilation = True or
#False).
#This experiment is performed on the small data subset.

#TODO: change NUM_EPOCHS to 100 to replicate paper results.
NUM_EPOCHS = 1 #temporarily set to 1 for fast code demo

if __name__=='__main__':
    lambda_val = 1.0/3.0
    for dilate in [True,False]:
        for downsamp_mode in ['nearest','area','trilinear']:
            tot0 = timeit.default_timer()
            #DS stands for Downsampling Study
            DukeCTExperiment(descriptor='AxialNetMask_SmallData_dilate'+str(dilate)+'_'+downsamp_mode,
                custom_net = custom_models_mask.AxialNet_Mask,
                custom_net_args = {'n_outputs':80},
                loss_string = 'AxialNet_Mask-loss',
                loss_args = {'lambda_val':lambda_val},
                learning_rate = 1e-3, #default 1e-3
                weight_decay = 1e-7, #default 1e-7
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
                                        'num_channels':3,
                                        'crop_type':'single',
                                        'selfsupervised':False,
                                        'from_seg':False},
                            'attn_gr_truth_prep_args':{
                                        'dilate':dilate,
                                        'downsamp_mode':downsamp_mode},
                            'selected_note_acc_files':{'train':'/home/rlb61/data/img-hiermodel2/load_dataset/ground_truth/predefined_subsets/2020-01-10-imgtrain_random2000.csv',
                                                       'valid':'/home/rlb61/data/img-hiermodel2/load_dataset/ground_truth/predefined_subsets/2020-01-10-imgvalid_a_random1000.csv'}})
            tot1 = timeit.default_timer()
            print('Total Time', round((tot1 - tot0)/60.0,2),'minutes')
