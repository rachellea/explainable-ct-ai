#main.py

import timeit

from run_experiment import DukeCTExperiment
from models import custom_models_mask
from load_dataset import custom_datasets

if __name__=='__main__':
    lambda_val = 1.0/3.0
    downsamp_mode = 'nearest'
    dilate = False
    
    tot0 = timeit.default_timer()
    DukeCTExperiment(descriptor='WHOLEDATA_BodyAvg_MaskLoss_VanillaGradCAM',
        custom_net = custom_models_mask.BodyAvg_Mask_VanillaGradCAM,
        custom_net_args = {'n_outputs':80}, #80 labels (now that I've removed heart_nodule)
        loss_string = 'BodyAvg_Mask-loss',
        loss_args = {'lambda_val':lambda_val},
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
                                #Use the attn_storage_dir that you used for the other mask loss model
                                #because the attn gr truth should be identical and you do not want to
                                #have to re-compute so much attn gr truth!
                                'attn_storage_dir':'/storage/rlb61-data/img-hiermodel2/results/2020-10/2020-10-08_WHOLEDATA_BodyAvg_Mask_dilateFalse_nearest_Lambda33_FreshStart/attn_storage_temp/',
                                'dilate':dilate,
                                'downsamp_mode':downsamp_mode},
                    'selected_note_acc_files':{'train':'','valid':''}})
    tot1 = timeit.default_timer()
    print('Total Time', round((tot1 - tot0)/60.0,2),'minutes')


