#main.py

import timeit

from ct_img_model import DukeCTModel
from custom_models import ResNet18_Batch
from custom_datasets import CTDataset_2019_10

if __name__=='__main__':
    for abnormality in ['consolidation', 'mass', 'pericardial_effusion',
                        'cardiomegaly', 'pneumothorax']:
        print('\n\n\n\n********** Working on',abnormality,'**********')
        tot0 = timeit.default_timer()
        DukeCTModel(descriptor='resnet18-small-83-Original3DConv-'+abnormality,
                    custom_net = ResNet18_Batch,
                    custom_net_args = {'n_outputs':1},
                    loss = 'bce', loss_args = {}, #or empty dict
                    num_epochs=100, patience = 15,
                    batch_size = 2, device = 'all', data_parallel = True,
                    use_test_set = False, task = 'train_eval',
                    old_params_dir = '',
                    dataset_class = CTDataset_2019_10,
                    dataset_args = {'label_type_ld':'disease_new',
                                        'label_meanings':[abnormality], #can be 'all' or a list of strings
                                        'num_channels':3,
                                        'pixel_bounds':[-1000,200],
                                        'data_augment':True,
                                        'crop_type':'single',
                                        'selected_note_acc_files':{'train':'/home/rlb61/data/img-hiermodel2/data/pace_labels/predefined_subsets/2020-01-10-imgtrain_random2000.csv',
                                            'valid':'/home/rlb61/data/img-hiermodel2/data/pace_labels/predefined_subsets/2020-01-10-imgvalid_a_random1000.csv'}}
                    )
        tot1 = timeit.default_timer()
        print('Total Time', round((tot1 - tot0)/60.0,2),'minutes')