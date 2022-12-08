#test_losses.py
#Copyright (c) 2020 Rachel Lea Ballantyne Draelos

#MIT License

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE

import os
import copy
import torch
import unittest
import numpy as np
import torch.nn as nn

from src.evals import losses
from src.load_dataset import custom_datasets
from src.load_dataset.vol_proc import mask

class TestMaskRelatedLossCode(unittest.TestCase):
    def test_if_torch_and_np_have_equivalent_flip_and_rotate(self):
        x_np = np.random.rand(50,50,50)
        x_tr = torch.Tensor(copy.deepcopy(x_np)).to(torch.device('cuda'))
        #Checks
        assert (np.isclose(np.rot90(np.flip(x_np,axis=1), k=2, axes=(1,2)),
                          (torch.rot90(torch.flip(x_tr, dims=(1,)),  k=2, dims=(1,2))).cpu().numpy())).all()
        
        assert (np.isclose(np.rot90(np.flip(x_np,axis=2), k=1, axes=(1,2)),
                          (torch.rot90(torch.flip(x_tr, dims=(2,)),  k=1, dims=(1,2))).cpu().numpy())).all()
        
        assert (np.isclose(np.rot90(np.flip(x_np,axis=0), k=0, axes=(1,2)),
                          (torch.rot90(torch.flip(x_tr, dims=(0,)),  k=0, dims=(1,2))).cpu().numpy())).all()
        print('Passed check_if_torch_and_np_have_equivalent_flip_and_rotate()')
    
def visualize_flip_and_rotate_attn_gr_truth():
    """This is not a unit test. You have to call this function manually
    and separately e.g. in an interpreter. This function makes visualizations
    based on the attention ground truth to enable manual verification
    that the attention ground truth was created reasonably for all of the
    diseases for one scan. In order to run this function you must be able
    to make a valid call to custom_datasets.CTDataset_2019_10()"""
    print('Working on visualize_flip_and_rotate_attn_gr_truth()')
    main_results_dir = './temp_testing_attn_gr_truth'
    if not os.path.exists(main_results_dir):
        os.mkdir(main_results_dir)
    for dilate in [True,False]:
        for downsamp_mode in ['nearest','trilinear','area']:
            attn_storage_dir = os.path.join(main_results_dir,'dilate'+str(dilate)+'_'+downsamp_mode)
            if not os.path.exists(attn_storage_dir):
                os.mkdir(attn_storage_dir)
            dataset_args = {'verbose':False,
                            'label_type_ld':'location_disease_0323',
                            'genericize_lung_labels':True,
                            'label_counts':{'mincount_heart':200, #default 200
                                        'mincount_lung':125}, #default 125
                            'view':'axial',
                            'use_projections9':True,
                            'loss_string':'BodyAvg_Mask-loss',
                            'volume_prep_args':{
                                        'pixel_bounds':[-1000,800], #Recent increase in upper pixel bound to 800 -> better perf
                                        'num_channels':3,
                                        'crop_type':'single',
                                        'selfsupervised':False,
                                        'from_seg':False},
                            'attn_gr_truth_prep_args':{
                                        'attn_storage_dir':attn_storage_dir,
                                        'dilate':dilate,
                                        'downsamp_mode':downsamp_mode},
                            'selected_note_acc_files':{'train':'/home/rlb61/data/img-hiermodel2/load_dataset/ground_truth/predefined_subsets/2020-01-10-imgtrain_random2000.csv',
                                                       'valid':'/home/rlb61/data/img-hiermodel2/load_dataset/ground_truth/predefined_subsets/2020-01-10-imgvalid_a_random1000.csv'}}
            dataset = custom_datasets.CTDataset_2019_10(setname='valid',**dataset_args)
            label_meanings = dataset.return_label_meanings() #already genericized
            n_outputs_lung = len([x for x  in label_meanings if 'lung' in x])
            n_outputs_heart = len(label_meanings)-n_outputs_lung
            device = torch.device('cpu')
            
            #Pick a sample above the mean (~10) in terms of total number of
            #abnormalities. Note that <sample> has no batch dimension,
            #because it was NOT obtained using a DataLoader (just a Dataset)
            for idx in range(300):
                temp_sample = dataset[idx]
                total_abns = int(torch.sum(temp_sample['gr_truth']))
                if total_abns > 16:
                    sample = temp_sample
                    print('Found a sample! It has',total_abns,'abnormalities!')
                    break
                        
            #The code in mask.make_gifs_of_mask() assumes that we have a sample
            #without a batch dimension, because it is applied in the data
            #processing stage. However, the code in
            #losses.flip_and_rotate_attn_gr_truth() assumes that we DO have a
            #batch dimension, for the attn_gr_truth and for the auglabel.
            batch = {}
            batch['attn_gr_truth'] = sample['attn_gr_truth'].unsqueeze(0)
            batch['auglabel'] = sample['auglabel'].unsqueeze(0)
            attn_gr_truth = losses.flip_and_rotate_attn_gr_truth(batch, device)
            disease_masks = {}
            for label_idx in range(len(label_meanings)):
                #First get the key label_name, which is the disease
                #name (e.g. 'pneumonia') along with a specification of
                #where the disease is present, or whether is absent:
                label_name = label_meanings[label_idx]
                if label_idx < n_outputs_heart:
                    if sample['heart_gr_truth'][label_idx] == 1:
                        label_name += '_present'
                    else:
                        label_name += '_absent'
                else:
                    adj_label_idx = label_idx - n_outputs_heart
                    if sample['left_lung_gr_truth'][adj_label_idx] == 1:
                        label_name += '_LLpresent'
                    if sample['right_lung_gr_truth'][adj_label_idx] == 1:
                        label_name += '_RLpresent'
                    if (sample['left_lung_gr_truth'][adj_label_idx]+sample['right_lung_gr_truth'][adj_label_idx]) == 0:
                        label_name += '_absent'
                #Now that you have the key, fill in the value:
                upsample_shape = [sample['data'].shape[0]*sample['data'].shape[1],sample['data'].shape[2],sample['data'].shape[3]]
                disease_masks[label_name] = nn.functional.interpolate(attn_gr_truth[label_idx,:,:,:].unsqueeze(0).unsqueeze(0),
                                                                      size=upsample_shape,
                                                                      mode=downsamp_mode).squeeze()
            #Finally, make visualizations:
            mask.make_gifs_of_mask('attn_gr_truth', sample, disease_masks, attn_storage_dir,
              mask_orientation='transformed')
    print('See ./temp_testing_attn_gr_truth for attn_gr_truth visualizations')

if __name__=='__main__':
    unittest.main()
