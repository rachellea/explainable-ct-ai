#main.py

import timeit

from run_experiment import DukeCTExperiment
from models import custom_models_base
from load_dataset import custom_datasets

if __name__=='__main__':
    tot0 = timeit.default_timer()
    DukeCTExperiment(descriptor='BodyAvg800',
                custom_net = custom_models_base.Body_Avg,
                custom_net_args = {'n_outputs':132},
                loss_string = 'bce',
                learning_rate = 1e-3, #default 1e-3
                weight_decay = 1e-7, #default 1e-7
                num_epochs=100, patience = 15,
                batch_size = 1, device = 0,
                data_parallel = False, model_parallel = False,
                use_test_set = False, task = 'train_eval',
                old_params_dir = '',
                dataset_class = custom_datasets.CTDataset_2019_10,
                which_labels = 'all', #can be 'all' or 'fliprot'
                dataset_args = {'label_type_ld':'location_disease_0323',
                                'label_counts':{'mincount_heart':200, #default 200
                                                'mincount_lung':125}, #default 125
                                'view':'axial',
                                'projections9':True,
                                'data_augment':{'train':True,
                                                'valid':False,#NO DATA AUG ON VALIDATION SET IN SELFSUP NOW
                                                'test':False},
                                'volume_prep_args':{
                                            'pixel_bounds':[-1000,800],#HIGHER UPPER LIMIT OF PIXEL VALUES
                                            'max_slices':45,
                                            'max_side_length':420,
                                            'num_channels':3,
                                            'crop_type':'single',
                                            'selfsupervised':False,
                                            'from_seg':False},
                                'selected_note_acc_files':{'train':'/home/rlb61/data/img-hiermodel2/load_dataset/ground_truth/predefined_subsets/2020-01-10-imgtrain_random2000.csv',
                                                           'valid':'/home/rlb61/data/img-hiermodel2/load_dataset/ground_truth/predefined_subsets/2020-01-10-imgvalid_a_random1000.csv'}
                                })
    
    tot1 = timeit.default_timer()
    print('Total Time', round((tot1 - tot0)/60.0,2),'minutes')