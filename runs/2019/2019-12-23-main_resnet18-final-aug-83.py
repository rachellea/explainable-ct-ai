#main.py

import timeit

from ct_img_model import DukeCTModel
from custom_models import ResNet18_Batch
from custom_datasets import CTDataset_2019_10

if __name__=='__main__':
    tot0 = timeit.default_timer()
    DukeCTModel(descriptor='resnet18-final-aug-83',
                custom_net = ResNet18_Batch,
                custom_net_args = {'n_outputs':83},
                num_epochs=100,
                patience = 15, num_workers = 8, batch_size = 2,
                device = 'all', loss = 'bce',
                loss_args = {}, #or empty dict
                dataset_class = CTDataset_2019_10,
                dataset_args = {'label_type_ld':'disease_new',
                                    'label_meanings':'all', #can be 'all' or a list of strings
                                    'num_channels':3,
                                    'pixel_bounds':[-1000,200],
                                    'data_augment':True,
                                    'crop_type':'single'},
                data_parallel = True)
    tot1 = timeit.default_timer()
    print('Total Time', round((tot1 - tot0)/60.0,2),'minutes')