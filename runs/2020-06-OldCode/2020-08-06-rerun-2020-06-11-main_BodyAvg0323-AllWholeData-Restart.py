#main.py

import timeit

from ct_img_model import DukeCTModel
from models import custom_models_base
from load_dataset import custom_datasets

if __name__=='__main__':
    tot0 = timeit.default_timer()
    DukeCTModel(descriptor='BodyAvgAllWholeData',
                custom_net = custom_models_base.BodyAvg,
                custom_net_args = {'n_outputs':83, 'slices':45},
                loss = 'bce', loss_args = {},
                num_epochs=100, patience = 15,
                batch_size = 1, device = 0,
                data_parallel = False, model_parallel = False,
                use_test_set = False, task = 'restart_train_eval',
                old_params_dir = '/home/rlb61/data/OLD-IMG-HIERMODEL/results/2020-07-31_BodyAvgAllWholeData/params/',
                dataset_class = custom_datasets.CTDataset_2019_10,
                dataset_args = {'label_type_ld':'location_disease_0323',
                                    'label_meanings':'all', #can be 'all' or a list of strings
                                    'num_channels':3,
                                    'pixel_bounds':[-1000,200],
                                    'max_slices':135, #no projections since they aren't all saved locally
                                    'max_side_length':420,
                                    'data_augment':True,
                                    'crop_type':'single',
                                    'selected_note_acc_files':{'train':'','valid':''},
                                    'from_seg':False,
                                    'view':'axial',
                                    'projections9':False}) #no projections since they aren't all saved locally
    tot1 = timeit.default_timer()
    print('Total Time', round((tot1 - tot0)/60.0,2),'minutes')