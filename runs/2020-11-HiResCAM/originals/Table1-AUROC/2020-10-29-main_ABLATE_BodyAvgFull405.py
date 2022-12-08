#main.py

import timeit

from run_experiment import DukeCTExperiment
from models import custom_models_ablation
from load_dataset import custom_datasets

#Small data subset of 2,000 and 1,000 volumes
#But train it on full size data, so that you can get a comparison of the
#Mask Loss trained on full size data with the Old CTNet model trained on
#full size data

if __name__=='__main__':
    experiments = {'BodyAvg_FinalMaxPool':custom_models_ablation.Ablate_BodyAvg_FinalMaxPool,
                   'BodyAvg_NoCustomConv':custom_models_ablation.Ablate_BodyAvg_NoCustomConv,
                   'BodyAvg_SigmoidAfterCustomConv':custom_models_ablation.Ablate_BodyAvg_SigmoidAfterCustomConv,
                   'BodyAvg_RandomInitResNet':custom_models_ablation.Ablate_BodyAvg_RandomInitResNet}
    
    for model_description in['BodyAvg_SigmoidAfterCustomConv','BodyAvg_RandomInitResNet','BodyAvg_FinalMaxPool','BodyAvg_NoCustomConv']:
        chosen_model = experiments[model_description]
        
        tot0 = timeit.default_timer()
        #'ABL' stands for 'ablation'
        #'Update' applies although it is not part of the name
        DukeCTExperiment(descriptor='ABLATE_'+str(model_description)+'_Full405',
            custom_net = chosen_model,
            custom_net_args = {'n_outputs':80}, #80 labels (now that I've removed heart_nodule)
            loss_string = 'bce',
            loss_args = {'lambda_val':None},
            learning_rate = 1e-3, #default 1e-3
            weight_decay = 1e-7, #default 1e-7
            num_epochs=100, patience = 15,
            batch_size = 1, device = 0,
            data_parallel = False, model_parallel = False,
            use_test_set = False, task = 'train_eval',
            old_params_dir = '',
            dataset_class = custom_datasets.CTDataset_2019_10,
            dataset_args = {
                        'label_type_ld':'location_disease_0323',
                        'genericize_lung_labels':True,
                        'label_counts':{'mincount_heart':200, #default 200
                                    'mincount_lung':125}, #default 125
                        'view':'axial',
                        'use_projections9':False,
                        'volume_prep_args':{
                                    'pixel_bounds':[-1000,800], #Recent increase in upper pixel bound to 800 -> better perf
                                    'num_channels':3,
                                    'crop_type':'single',
                                    'selfsupervised':False,
                                    'from_seg':False},
                        'attn_gr_truth_prep_args':{
                                    'dilate':False,
                                    'downsamp_mode':False},
                        'selected_note_acc_files':{'train':'/home/rlb61/data/img-hiermodel2/load_dataset/ground_truth/predefined_subsets/2020-01-10-imgtrain_random2000.csv',
                                                   'valid':'/home/rlb61/data/img-hiermodel2/load_dataset/ground_truth/predefined_subsets/2020-01-10-imgvalid_a_random1000.csv'}
                                    })
        tot1 = timeit.default_timer()
        print('Total Time', round((tot1 - tot0)/60.0,2),'minutes')
