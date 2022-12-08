#main.py

import timeit

from run_experiment import DukeCTExperiment
from models import custom_models_old
from load_dataset import custom_datasets

if __name__=='__main__':
    tot0 = timeit.default_timer()
    DukeCTExperiment(descriptor='WHOLEDATA_OldCTNet_0323Disease83Labels',
        custom_net = custom_models_old.ResNet18_Original,
        custom_net_args = {'n_outputs':83,'slices':135}, #80 labels (now that I've removed heart_nodule)
        loss_string = 'bce',
        loss_args = {},
        learning_rate = 1e-3, #default 1e-3
        weight_decay = 1e-7, #default 1e-7
        num_epochs=100, patience = 15,
        batch_size = 1, device = 0,
        data_parallel = False, model_parallel = False,
        use_test_set = False, task = 'train_eval',
        old_params_dir = '',
        dataset_class = custom_datasets.CTDataset_2019_10,
        dataset_args = {
                    'label_type_ld':'disease_0323',
                    'genericize_lung_labels':False,
                    'label_counts':{'mincount_heart':None, #default 200
                                'mincount_lung':None}, #default 125
                    'view':'axial',
                    'use_projections9':False,
                    'volume_prep_args':{
                                'pixel_bounds':[-1000,800], #Recent increase in upper pixel bound to 800 -> better perf
                                'num_channels':3,
                                'crop_type':'single',
                                'selfsupervised':False,
                                'from_seg':False},
                    'attn_gr_truth_prep_args':{
                                'dilate':None,
                                'downsamp_mode':None},
                    'selected_note_acc_files':{'train':'','valid':''}})
    tot1 = timeit.default_timer()
    print('Total Time', round((tot1 - tot0)/60.0,2),'minutes')
