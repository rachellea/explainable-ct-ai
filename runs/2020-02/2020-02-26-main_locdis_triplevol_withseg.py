#main.py

import timeit

from ct_img_model import DukeCTModel
from models import custom_models_loc
from load_dataset import custom_datasets

if __name__=='__main__':
    tot0 = timeit.default_timer()
    DukeCTModel(descriptor='locdis_triplevol_withseg',
                custom_net = custom_models_loc.ResNet18_TripleCrop_Seg,
                custom_net_args = {'n_outputs_lung':28,'n_outputs_heart':27},
                loss = 'bce', loss_args = {}, #or empty dict
                num_epochs=100, patience = 15,
                batch_size = 1, device = 'all', data_parallel = True,
                use_test_set = False, task = 'train_eval',
                old_params_dir = '',
                dataset_class = custom_datasets.CTDataset_2019_10,
                dataset_args = {'label_type_ld':'location_disease_1',
                                    'label_meanings':'all', #can be 'all' or a list of strings
                                    'num_channels':3,
                                    'pixel_bounds':[-1000,200],
                                    'data_augment':True,
                                    'crop_type':'triple',
                                    'selected_note_acc_files':{'train':'/home/rlb61/data/img-hiermodel2/load_dataset/ground_truth/predefined_subsets/2020-01-10-imgtrain_random2000.csv',
                                        'valid':'/home/rlb61/data/img-hiermodel2/load_dataset/ground_truth/predefined_subsets/2020-01-10-imgvalid_a_random1000.csv'},
                                    'from_seg':True}
                )
    tot1 = timeit.default_timer()
    print('Total Time', round((tot1 - tot0)/60.0,2),'minutes')
