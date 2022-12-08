#main.py

import timeit

from ct_img_model import DukeCTModel
from custom_models import ResNet18_Batch
from custom_datasets import CTDataset_2019_10

if __name__=='__main__':
    tot0 = timeit.default_timer()
    DukeCTModel(descriptor='resnet18-final-aug-9',
                custom_net = ResNet18_Batch,
                custom_net_args = {'n_outputs':9},
                loss = 'bce', loss_args = {}, #or empty dict
                num_epochs=100, patience = 15,
                batch_size = 2,  device = 'all', data_parallel = True,
                use_test_set = True, task = 'predict_on_test', #task can also be 'train_eval'
                old_params_dir = '/home/rlb61/data/img-hiermodel2/results/2019-12-24_resnet18-final-aug-9/params/', #only needed if task is 'predict_on_test'
                dataset_class = CTDataset_2019_10,
                dataset_args = {'label_type_ld':'disease_new',
                                    'label_meanings':['nodule','opacity','atelectasis','pleural_effusion','consolidation','mass','pericardial_effusion','cardiomegaly','pneumothorax'],
                                    'num_channels':3,
                                    'pixel_bounds':[-1000,200],
                                    'data_augment':True,
                                    'crop_type':'single',
                                    'selected_note_acc_files':{'train':'','valid':'','test':''}},
                )
    tot1 = timeit.default_timer()
    print('Total Time', round((tot1 - tot0)/60.0,2),'minutes')