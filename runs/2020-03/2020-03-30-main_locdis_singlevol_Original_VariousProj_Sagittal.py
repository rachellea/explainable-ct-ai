#main.py

import timeit

from ct_img_model import DukeCTModel
from models import custom_models
from load_dataset import custom_datasets2

if __name__=='__main__':
    tot0 = timeit.default_timer()
    n_pool_options = [3,1,5]
    for n_pool in n_pool_options:
        DukeCTModel(descriptor='locdis_singlevol_Original_Proj'+str(n_pool)+'_Sagittal',
                custom_net = custom_models.ResNet18_Original_Proj_Slices,
                custom_net_args = {'n_outputs':83, 'n_slices':402, 'n_pool':n_pool},
                loss = 'bce', loss_args = {}, #or empty dict
                num_epochs=100, patience = 15,
                batch_size = 1, device = 0,
                data_parallel = False, model_parallel = False,
                use_test_set = False, task = 'train_eval',
                old_params_dir = '',
                dataset_class = custom_datasets2.CTDataset_2019_10,
                dataset_args = {'label_type_ld':'location_disease_1',
                                    'label_meanings':'all', #can be 'all' or a list of strings
                                    'num_channels':1,
                                    'pixel_bounds':[-1000,200],
                                    'max_slices':402,
                                    'max_side_length':420,
                                    'data_augment':True,
                                    'crop_type':'single',
                                    'selected_note_acc_files':{'train':'/home/rlb61/data/img-hiermodel2/load_dataset/ground_truth/predefined_subsets/2020-01-10-imgtrain_random2000.csv',
                                        'valid':'/home/rlb61/data/img-hiermodel2/load_dataset/ground_truth/predefined_subsets/2020-01-10-imgvalid_a_random1000.csv'},
                                    'from_seg':False,
                                    'view':'sagittal'})
    tot1 = timeit.default_timer()
    print('Total Time', round((tot1 - tot0)/60.0,2),'minutes')