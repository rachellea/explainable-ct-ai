#test_custom_datasets_text.py
#Copyright (c) 2021 Rachel Lea Ballantyne Draelos

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

#for unit test:
import unittest
import numpy as np

#for visualization:
import torch
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.ioff()

from src.load_dataset import custom_datasets_text

class TestRemoveFrame(unittest.TestCase):
    def test_remove_residual_frame_of_ones(self):
        #Repair a column of all ones
        input1 = np.array([[0,0,0,1],
                           [0,0,1,1],
                           [0,0,1,1],
                           [1,0,1,1]], dtype='bool')
        output1 = custom_datasets_text.remove_residual_frame_of_ones(input1)
        correct1 = np.array([[0,0,0,0],
                             [0,0,1,0],
                             [0,0,1,0],
                             [1,0,1,0]], dtype='bool')
        assert (output1 == correct1).all()
        
        #Repair two rows of all ones
        input2 = np.array([[1,1,0,0],
                           [0,0,1,1],
                           [1,1,1,1],
                           [1,1,1,1]], dtype='bool')
        output2 = custom_datasets_text.remove_residual_frame_of_ones(input2)
        correct2 = np.array([[1,1,0,0],
                             [0,0,1,1],
                             [0,0,0,0],
                             [0,0,0,0]], dtype='bool')
        assert (output2 == correct2).all()
        
        #Repair both a column and a row of all ones
        input3 = np.array([[0,0,0,1],
                           [1,0,1,1],
                           [0,0,1,1],
                           [1,1,1,1]], dtype='bool')
        output3 = custom_datasets_text.remove_residual_frame_of_ones(input3)
        correct3 = np.array([[0,0,0,0],
                             [1,0,1,0],
                             [0,0,1,0],
                             [0,0,0,0]], dtype='bool')
        assert (output3 == correct3).all()
        
        #Do nothing to the array; it is already fine
        input4 = np.array([[0,0,0,0],
                           [0,0,1,0],
                           [0,0,1,1],
                           [1,0,1,1]], dtype='bool')
        output4 = custom_datasets_text.remove_residual_frame_of_ones(input4)
        assert (output4 == input4).all()
        print('Passed test_remove_residual_frame_of_ones()')

#The following function needs to be called manually if you want to make
#the visualizations to sanity check the superimposition of text in
#grayscale over the CT slice. In an interpreter, run these commands:
#>>> import tests.load_dataset.test_custom_datasets_text as cdt
#>>> cdt.make_viz_for_text_over_ct_slice()
#You can also uncomment the mask visualization code within custom_datasets_text.py
#if you want to see visualizations of the masks too.
def make_viz_for_text_over_ct_slice():
    dataset_args = {'setname':'valid',
                    'verbose':False,
                    'loss_string':'bce',
                        'label_type_ld':'location_disease_0323',
                        'genericize_lung_labels':True,
                        'label_counts':{'mincount_heart':200,
                                    'mincount_lung':125},
                        'view':'axial',
                        'use_projections9':False,
                        'volume_prep_args':{
                                    'pixel_bounds':[-1000,800],
                                    'num_channels':1,
                                    'crop_type':'single',
                                    'selfsupervised':False,
                                    'from_seg':False},
                        'attn_gr_truth_prep_args':{
                                'dilate':None,
                                'downsamp_mode':None},
                        #Paths
                        'selected_note_acc_files':{'train':'/home/rlb61/data/img-hiermodel2/data/RADChestCT_DEID/predefined_subsets/2020-01-10-imgtrain_random2000_DEID.csv',
                                                   'valid':'/home/rlb61/data/img-hiermodel2/data/RADChestCT_DEID/predefined_subsets/2020-01-10-imgvalid_a_random1000_DEID.csv'},
                        'ct_scan_path':'/scratch/rlb61/2019-10-BigData-DEID',
                        'ct_scan_projections_path':'/scratch/rlb61/2020-04-15-Projections-DEID',
                        'key_csvs_path':'/home/rlb61/data/img-hiermodel2/data/RADChestCT_DEID/',
                        'segmap_path':'/storage/rlb61-ct_images/vna/rlb61/2019-10-BigData-segmaps-DEID/'}
    dataset = custom_datasets_text.CTDataset_2019_10_with_Text(**dataset_args)
    sample = dataset[0]
    torch.save(sample['data'],'data.pt')
    
    for idx in range(200,250):
        if sample['textlabel'][idx] == 1:
            plt.figure(figsize=(8, 8))
            plt.imshow(sample['data'].numpy()[idx,:,:], cmap = plt.cm.gray)
            plt.tight_layout(pad=0)
            plt.gca().set_axis_off()
            plt.savefig('ct_w_text_slice_'+str(idx)+'.png', bbox_inches='tight',pad_inches=0)
            plt.close()

if __name__ == '__main__':
    unittest.main()