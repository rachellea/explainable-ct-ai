from viz_attn_2d_slices import *

if __name__=='__main__':
    tot0 = timeit.default_timer()
    m = MakeAttentionVisualizations(descriptor='BodyAvg_SpatialAtt3_4-sigmoid',
                experimentname = 'VizAttention2D',
                custom_net = custom_models_attn3.BodyAvg_SpatialAtt3_4,
                custom_net_args = {'n_outputs':83,'nonlinearity':'sigmoid'},
                loss = 'bce', loss_args = {}, #or empty dict
                num_epochs=100, patience = 15,
                batch_size = 1, device = 0,
                data_parallel = False, model_parallel = False,
                use_test_set = False, task = 'visualize',
                old_params_dir = '/home/rlb61/data/img-hiermodel2/results/2020-07-17_BodyAvg_SpatialAtt3_4-sigmoid/params/BodyAvg_SpatialAtt3_4-sigmoid',
                dataset_class = custom_datasets.CTDataset_2019_10,
                dataset_args = {'label_type_ld':'location_disease_0323',
                                'label_meanings':'all', #can be 'all' or a list of strings
                                'view':'axial',
                                'projections9':True, 
                                'data_augment':{'train':True,
                                            'valid':False,#normally False, except for self-supervised learning
                                            'test':False},
                                'volume_prep_args':{
                                            'pixel_bounds':[-1000,200],
                                            'max_slices':45,
                                            'max_side_length':420,
                                            'num_channels':3,
                                            'crop_type':'single',
                                            'selfsupervised':False,
                                            'from_seg':False},
                                'selected_note_acc_files':{'train':'/home/rlb61/data/img-hiermodel2/load_dataset/ground_truth/predefined_subsets/2020-01-10-imgtrain_random2000.csv',
                                                           'valid':'/home/rlb61/data/img-hiermodel2/load_dataset/ground_truth/predefined_subsets/2020-01-10-imgvalid_a_random1000.csv'}
                                })
    m.run_model()
    tot1 = timeit.default_timer()
    print('Total Time', round((tot1 - tot0)/60.0,2),'minutes')
