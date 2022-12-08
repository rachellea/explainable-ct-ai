#run6_AxialNet_test_wholedata.py
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

from models import custom_models_base
from load_dataset import custom_datasets
from run_experiment import DukeCTExperiment

#run6: Calculate the test set performance of the AxialNet model that was trained
#on the full data.

if __name__=='__main__':
    tot0 = timeit.default_timer()
    DukeCTExperiment(descriptor='AxialNet_WholeData',
        custom_net = custom_models_base.AxialNet,
        custom_net_args = {'n_outputs':80},
        loss_string = 'bce',
        loss_args = {},
        learning_rate = 1e-3,
        weight_decay = 1e-7,
        num_epochs=1, patience = 15,
        batch_size = 1, device = 0,
        data_parallel = False, model_parallel = False,
        use_test_set = True, task = 'predict_on_test', #Evaluate final model on the test set
        
        #TODO: replace this with the correct path to the params directory containing the params of the final model
        old_params_dir = '/home/rlb61/data/img-hiermodel2/results/2020-12-12_AxialNet_WholeData/params/', 
        
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
                                'dilate':None,
                                'downsamp_mode':None},
                    'selected_note_acc_files':{'train':'','valid':'','test':''}})
    tot1 = timeit.default_timer()
    print('Total Time', round((tot1 - tot0)/60.0,2),'minutes')
