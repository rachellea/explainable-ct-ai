#main.py

import timeit

from run_experiment import DukeCTExperiment
from load_dataset import custom_datasets
from models import custom_models_multiview


def merge_dicts(a,b):
    for key in a.keys():
        b[key] = a[key]
    return b

SHARED_ARGS =  {'learning_rate':1e-3, #default 1e-3
            'weight_decay':1e-7, #default 1e-7
            'num_epochs':100,
            'patience':15,
            'batch_size':1,
            'device':0,
            'data_parallel':False,
            'model_parallel':False,
            'use_test_set':False,
            'task':'train_eval',
            'old_params_dir':'',
            'dataset_class':custom_datasets.CTDataset_2019_10}

SHARED_DATASET_ARGS = {'label_type_ld':'disease_0323',
            'genericize_lung_labels':False,
            'label_counts':{'mincount_heart':None, 'mincount_lung':None},
            'include_mask':False}

################################################################################
################################################################################
################################################################################
if __name__=='__main__':
    tot0 = timeit.default_timer()
    conv = ['sharedconv','separateconv']
    func = ['weighted_avg','avg','max']
    for conv_type in conv:
        for func_type in func:
            DukeCTExperiment(descriptor='Tricks-Multiview-'+conv_type+'-'+func_type,
                        custom_net = custom_models_multiview.BodyAvgMultiview,
                        custom_net_args = {'n_outputs':83,'chosen_conv':conv_type,'chosen_comb_func':func_type},
                        loss_string = 'bce',
                        dataset_args = merge_dicts(SHARED_DATASET_ARGS,
                                        {'view':'all', #VIEW IS ALL SINCE THIS IS A MULTIVIEW MODEL***
                                        'projections9':True,
                                        'data_augment':{'train':True,'valid':False,'test':False},
                                        'volume_prep_args':{
                                                'pixel_bounds':[-1000,800],
                                                'max_slices':45, #45 WHEN USING 9PROJECTIONS WILL BECOME 15 IN THE MODEL B/C 3 CHANNELS
                                                'max_side_length':420, #UNCHANGED
                                                'num_channels':3,
                                                'crop_type':'single', #UNCHANGED
                                                'selfsupervised':False,
                                                'from_seg':False}, #UNCHANGED
                                        'selected_note_acc_files':{'train':'/home/rlb61/data/img-hiermodel2/load_dataset/ground_truth/predefined_subsets/2020-01-10-imgtrain_random2000.csv', #UNCHANGED
                                                           'valid':'/home/rlb61/data/img-hiermodel2/load_dataset/ground_truth/predefined_subsets/2020-01-10-imgvalid_a_random1000.csv'}}), #UNCHANGED
                **SHARED_ARGS)
    tot1 = timeit.default_timer()
    print('Total Time', round((tot1 - tot0)/60.0,2),'minutes')