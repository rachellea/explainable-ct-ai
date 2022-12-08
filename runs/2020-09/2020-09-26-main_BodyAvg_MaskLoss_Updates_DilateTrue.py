#main.py

import timeit

from run_experiment import DukeCTExperiment
from models import custom_models_mask
from load_dataset import custom_datasets

#Updates from the last time this model was run:
#(1) I got rid of heart_nodule, heart_mass, heart_cancer, and other heart
#    labels that don't make much sense.
#(2) I updated and tested the genericize lung labels function so that now if
#    a lung disease like pneumonia is listed as present in the lung but a
#    right lung or left lung is not specified, the right lung and left lung
#    will get marked as positive (=1) for that disease. This is different from
#    before where if a side was not specified in the ground truth labels, then
#    the right lung and left lung would be negative (=0). Making this update
#    should improve the results for the mask loss because with the mask loss
#    we're just saying where the model is and is not allowed to look, and so if
#    there is pneumonia and we don't know the side, we should allow the model
#    to look for it in either lung. (The reason that we were using the 0s before
#    is that I was using the right lung vs left lung as classification labels and
#    in that case I chose to make the label 0 if it wasn't explicitely stated
#    to be on the right side or left side.)
#(3) I also updated the mask loss in losses.py so that now if there is
#    a bad quality segmentation mask, instead of just skipping the mask loss
#    for that scan, it will instead calculate rough left lung, right lung,
#    and heart mini segmentation masks, and it will use those when making
#    the attention ground truth for that scan. This should also help the perf.
#(4) Finally, I got rid of random padding. Today I realized that the random
#    padding of up to 15 pixels has a potentially MASSIVE effect when
#    using precomputed projections, because the precomputed projections have
#    only 45 slices instead of 400-something. So, for this run, there is NO
#    RANDOM PADDING at all. But there is still data augmentation in the form of
#    random flips and/or random rotations.

#Note that I have updated custom_datasets.py and utils.py so that now in those
#files by default it does not do random padding. Also in mask.py it checks
#to ensure the sum of randpad6val is zero (i.e. it double checks that no random
#padding was applied)

if __name__=='__main__':
    for downsamp_mode in ['trilinear','area','nearest']:
        for dilate in [True]:
            tot0 = timeit.default_timer()
            #Put CORRECT in the title to distinguish from the few bad runs that
            #you did before back when there was a mistake in the attention
            #ground truth preparation
            DukeCTExperiment(descriptor='BodyAvg_Mask_CORRECT_dilate'+str(dilate)+'_'+downsamp_mode,
                custom_net = custom_models_mask.BodyAvg_Mask,
                custom_net_args = {'n_outputs':80}, #80 labels (now that I've removed heart_nodule)
                loss_string = 'BodyAvg_Mask-loss',
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
                            'use_projections9':True,
                            'volume_prep_args':{
                                        'pixel_bounds':[-1000,800], #Recent increase in upper pixel bound to 800 -> better perf
                                        'num_channels':3,
                                        'crop_type':'single',
                                        'selfsupervised':False,
                                        'from_seg':False},
                            'attn_gr_truth_prep_args':{
                                        'dilate':dilate,
                                        'downsamp_mode':downsamp_mode},
                            'selected_note_acc_files':{'train':'/home/rlb61/data/img-hiermodel2/load_dataset/ground_truth/predefined_subsets/2020-01-10-imgtrain_random2000.csv',
                                                       'valid':'/home/rlb61/data/img-hiermodel2/load_dataset/ground_truth/predefined_subsets/2020-01-10-imgvalid_a_random1000.csv'}})
    tot1 = timeit.default_timer()
    print('Total Time', round((tot1 - tot0)/60.0,2),'minutes')
    
    
    
    
    