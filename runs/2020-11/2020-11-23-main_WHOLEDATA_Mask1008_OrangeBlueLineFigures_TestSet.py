#main.py

import timeit

from models import custom_models_mask
from load_dataset import custom_datasets

from plot import viz_orangebluelines

if __name__=='__main__':
    tot0 = timeit.default_timer()
    m = viz_orangebluelines.MakeLineVisualizations(descriptor='WHOLEDATA_Mask1008_OrangeBlueLineFigures_TestSet',
                custom_net = custom_models_mask.BodyAvg_Mask,
                custom_net_args = {'n_outputs':80,'slices':135},
                batch_size = 1, device = 0,
                full_params_dir ='/storage/rlb61-data/img-hiermodel2/results/2020-10/2020-10-08_WHOLEDATA_BodyAvg_Mask_dilateFalse_nearest_Lambda33_FreshStart/params/WHOLEDATA_BodyAvg_Mask_dilateFalse_nearest_Lambda33_FreshStart_epoch4',
                dataset_class = custom_datasets.CTDataset_2019_10,
                dataset_args = {
                    'verbose':False,
                    'loss_string':'bce',
                        
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
                                'dilate':'',
                                'downsamp_mode':''},
                    'selected_note_acc_files':{'train':'','valid':'','test':''}},
                results_dir='/storage/rlb61-data/img-hiermodel2/results/2020-11-23_WHOLEDATA_Mask1008_OrangeBlueLineFigures_TestSet')
    m.run_model()
    tot1 = timeit.default_timer()
    print('Total Time', round((tot1 - tot0)/60.0,2),'minutes')
