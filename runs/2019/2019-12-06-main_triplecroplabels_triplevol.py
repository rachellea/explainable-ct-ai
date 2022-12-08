#main.py

import timeit

from ct_img_model import DukeCTModel
from custom_models import ResNet18_TripleCrop
from custom_datasets import CTDataset_2019_10

if __name__=='__main__':
    tot0 = timeit.default_timer()
    DukeCTModel(descriptor='triplecroplabels-triplevol',
                custom_net = ResNet18_TripleCrop,
                custom_net_args = {'n_outputs_lung':51, #102 total for right_lung and left_lung
                                   'n_outputs_heart':33}, #17 for 'heart' + 16 for 'great vessels' (h_)
                                   #total labels = 102 + 33 = 135
                num_epochs=100,
                patience = 15, num_workers = 44, batch_size = 1, #batch_size must be one for this model
                device = 3, loss = 'bce',
                loss_args = {}, #or empty dict
                dataset_class = CTDataset_2019_10,
                dataset_args = {'label_type_ld':'triple_crop_flat',
                                    'label_meanings':'all', #can be 'all' or a list of strings
                                    'num_channels':3,
                                    'pixel_bounds':[-1000,200],
                                    'data_augment':True,
                                    'crop_type':'triple'})
    tot1 = timeit.default_timer()
    print('Total Time', round((tot1 - tot0)/60.0,2),'minutes')