#2021-03-12-BodyConvBigger_Variant1.py

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

from src.run_experiment import DukeCTExperiment
from src.models import custom_models_bigger
from src.load_dataset import custom_datasets

if __name__=='__main__':
    experiments = {
        'BodyConvBigger_Variant1':{'custom_net':custom_models_bigger.BodyConvBigger_Variant1}}
    
    for key in experiments.keys():
        tot0 = timeit.default_timer()
        DukeCTExperiment(descriptor=key,
            base_results_dir = '/home/rlb61/data/img-hiermodel2/results/results_2021/',
            custom_net = experiments[key]['custom_net'],
            custom_net_args = {'n_outputs':80},
            loss_string = 'bce',
            loss_args = {},
            learning_rate = 1e-3,
            weight_decay = 1e-7,
            num_epochs=100, patience = 15,
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
                        'use_projections9':True,
                        'volume_prep_args':{
                                    'pixel_bounds':[-1000,800],
                                    'num_channels':3,
                                    'crop_type':'single',
                                    'selfsupervised':False,
                                    'from_seg':False},
                        'attn_gr_truth_prep_args':{
                                'dilate':None,
                                'downsamp_mode':None},
                        #Paths
                        'selected_note_acc_files':{'train':'/home/rlb61/data/img-hiermodel2/data/RADChestCT_DEID/predefined_subsets/2020-01-10-imgtrain_random2000_DEID.csv',
                                                   'valid':'/home/rlb61/data/img-hiermodel2/data/RADChestCT_DEID/predefined_subsets/2020-01-10-imgvalid_a_random1000_DEID.csv'},
                        'ct_scan_path':'/scratch/rlb61/2019-10-BigData-DEID',
                        'ct_scan_projections_path':'/scratch/rlb61/2020-04-15-Projections-DEID',
                        'key_csvs_path':'/home/rlb61/data/img-hiermodel2/data/RADChestCT_DEID/',
                        'segmap_path':'/storage/rlb61-ct_images/vna/rlb61/2019-10-BigData-segmaps-DEID/'})
        tot1 = timeit.default_timer()
        print('Total Time', round((tot1 - tot0)/60.0,2),'minutes')
