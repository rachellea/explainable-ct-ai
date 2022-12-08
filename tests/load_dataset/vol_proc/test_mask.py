#test_mask.py
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

import torch
import unittest
import numpy as np

from src.load_dataset.vol_proc import mask

class TestMaskCreationAndProcessing(unittest.TestCase):
    def test_create_rough_seg_masks_from_scratch(self):
        for n_slices in [3,8,23]:
            heart_6x6 = torch.Tensor(np.array([[0,1,1,1,1,0],
                                               [0,1,1,1,1,0],
                                               [0,1,1,1,1,0],
                                               [0,1,1,1,1,0],
                                               [0,1,1,1,1,0],
                                               [0,1,1,1,1,0]]))
            heart_segcorrect = torch.stack([heart_6x6]*n_slices,dim=0)
            
            right_lung_6x6 = torch.Tensor(np.array([[1,1,1,0,0,0],
                                                    [1,1,1,0,0,0],
                                                    [1,1,1,0,0,0],
                                                    [1,1,1,0,0,0],
                                                    [1,1,1,0,0,0],
                                                    [1,1,1,0,0,0]]))
            right_lung_segcorrect = torch.stack([right_lung_6x6]*n_slices,dim=0)
            
            left_lung_6x6 = torch.Tensor(np.array([[0,0,0,1,1,1],
                                                   [0,0,0,1,1,1],
                                                   [0,0,0,1,1,1],
                                                   [0,0,0,1,1,1],
                                                   [0,0,0,1,1,1],
                                                   [0,0,0,1,1,1]]))
            left_lung_segcorrect = torch.stack([left_lung_6x6]*n_slices,dim=0)
            
            heart_segout, right_lung_segout, left_lung_segout = mask.create_rough_seg_masks_from_scratch(slices=n_slices)
            
            assert (heart_segout==heart_segcorrect).all()
            assert (right_lung_segout==right_lung_segcorrect).all()
            assert (left_lung_segout==left_lung_segcorrect).all()
        print('Passed test_create_rough_seg_masks_from_scratch()')
    
    def test_shrink_the_seg_masks(self):
        """This isn't a perfect unit test because I haven't manually calculated
        the result of applying various downsampling algorithms on a realistic organ
        segmentation mask.
        However, it does test the following:
            * whether something that is all ones get sampled down to all ones,
              for all the available downsampling algorithms
            * whether something that is all zeroes gets sampled down to all zeros,
              for all the available downsampling algorithms
            * whether the assignment of 0 to background, 1 to right lung, 2 for
              heart, and 3 for left lung is working properly
            * whether the specification of the number of slices in the small mask
              works properly, and whether you can choose different numbers of slices
              successfully"""
        #0 for background, 1 for right lung, 2 for heart, and 3 for left lung
        full_size_masks = {'zeros':torch.zeros(321,406,234), #all background
                        'ones':torch.ones(123,432,300), #all right lung
                        'twos':2*torch.ones(200,300,400), #all heart
                        'threes':3*torch.ones(324,333,309)} #all left lung
        for downsamp_mode in [ 'nearest','trilinear','area']:
            for n_slices in [2,45,19]:
                correct_small_masks = {'zeros':{'heart_segcorrect': torch.zeros(n_slices,6,6),
                                                'right_lung_segcorrect': torch.zeros(n_slices,6,6),
                                                'left_lung_segcorrect': torch.zeros(n_slices,6,6)},
                                       'ones':{'heart_segcorrect':torch.zeros(n_slices,6,6),
                                               'right_lung_segcorrect':torch.ones(n_slices,6,6),
                                               'left_lung_segcorrect':torch.zeros(n_slices,6,6)},
                                       'twos':{'heart_segcorrect':torch.ones(n_slices,6,6),
                                               'right_lung_segcorrect':torch.zeros(n_slices,6,6),
                                               'left_lung_segcorrect':torch.zeros(n_slices,6,6)},
                                       'threes':{'heart_segcorrect':torch.zeros(n_slices,6,6),
                                                 'right_lung_segcorrect':torch.zeros(n_slices,6,6),
                                                 'left_lung_segcorrect':torch.ones(n_slices,6,6)}}
                for mask_type in full_size_masks.keys():
                    full_size_mask = full_size_masks[mask_type]
                    heart_segout, right_lung_segout, left_lung_segout = mask.shrink_the_seg_masks(full_size_mask, n_slices, downsamp_mode)
                    assert (heart_segout==correct_small_masks[mask_type]['heart_segcorrect']).all()
                    assert (right_lung_segout==correct_small_masks[mask_type]['right_lung_segcorrect']).all()
                    assert (left_lung_segout==correct_small_masks[mask_type]['left_lung_segcorrect']).all()
        print('Passed test_shrink_the_seg_masks()')
    
    def test_dilatemask(self):
        #See 2020-09-26-Unit-Test-Attn-Gr-Truth-Code/test_dilate_small_mask.pptx
        #Also, I made a bunch of visualizations of the result of dilation,
        #upsampled again and superimposed over the original CT, so I have a sense
        #of what it does on real data and it's reasonable.
        small_mask = torch.zeros(3,3,3)
        small_mask[1,1,1] = 1
        assert (mask.dilate_small_mask(small_mask)==torch.ones(3,3,3)).all()
        
        small_mask2 = torch.zeros(5,5,5)
        small_mask2[:,2,2] = 1
        small_mask2_dilated_correct = torch.zeros(5,5,5)
        small_mask2_dilated_correct[:,1:4,1:4] = 1
        assert (mask.dilate_small_mask(small_mask2)==small_mask2_dilated_correct).all()
        
        small_mask3 = torch.zeros(2,8,8)
        small_mask3[:,2,1] = 1
        small_mask3[:,6:,6:] = 1
        small_mask3_dilated_correct = torch.zeros(2,8,8)
        small_mask3_dilated_correct[:,1:4,0:3] = 1
        small_mask3_dilated_correct[:,5:,5:] = 1
        assert (mask.dilate_small_mask(small_mask3)==small_mask3_dilated_correct).all()
        print('Passed test_dilatemask()')
    
    def test_construct_attn_gr_truth(self):
        #First test
        zeros_seg_small = torch.zeros(15,6,6)
        heart_6x6 = torch.Tensor(np.array([[0,1,1,1,1,0],
                                           [0,1,1,1,1,0],
                                           [0,1,1,1,1,0],
                                           [0,1,1,1,1,0],
                                           [0,1,1,1,1,0],
                                           [0,1,1,1,1,0]]))
        heart_seg_small = torch.stack([heart_6x6]*15,dim=0)
        right_lung_6x6 = torch.Tensor(np.array([[1,1,1,0,0,0],
                                                [1,1,1,0,0,0],
                                                [1,1,1,0,0,0],
                                                [1,1,1,0,0,0],
                                                [1,1,1,0,0,0],
                                                [1,1,1,0,0,0]]))
        right_lung_seg_small = torch.stack([right_lung_6x6]*15,dim=0)
        left_lung_6x6 = torch.Tensor(np.array([[0,0,0,1,1,1],
                                               [0,0,0,1,1,1],
                                               [0,0,0,1,1,1],
                                               [0,0,0,1,1,1],
                                               [0,0,0,1,1,1],
                                               [0,0,0,1,1,1]]))
        left_lung_seg_small = torch.stack([left_lung_6x6]*15,dim=0)
        both_lung_seg_small = torch.ones(15,6,6)
        
        sample = {}
        sample['heart_gr_truth'] = np.array([0,1])
        sample['left_lung_gr_truth'] = np.array([0,1,1,0,1])
        sample['right_lung_gr_truth'] = np.array([1,1,0,0,1])
        attn_gr_truth_out = mask.construct_attn_gr_truth(sample, heart_seg_small, left_lung_seg_small, right_lung_seg_small)
        attn_gr_truth_correct = torch.stack([zeros_seg_small,heart_seg_small,
                                             right_lung_seg_small,both_lung_seg_small,
                                             left_lung_seg_small,zeros_seg_small,
                                             both_lung_seg_small],dim=0)
        for disease_idx in range(7):
            assert (attn_gr_truth_out[disease_idx,:,:,:]==attn_gr_truth_correct[disease_idx,:,:,:]).all()
        assert (attn_gr_truth_out==attn_gr_truth_correct).all()
        
        #Second test - right lung and left lung masks overlap in one place and leave
        #a gap in another place. The heart and lung masks are both irregular.
        heart_6x6_2 = torch.Tensor(np.array([[0,0,0,1,0,0],
                                           [0,1,1,0,1,0],
                                           [0,1,1,1,1,0],
                                           [0,0,1,1,1,0],
                                           [0,0,1,1,0,0],
                                           [0,0,0,0,0,0]]))
        heart_seg_small2 = torch.stack([heart_6x6_2]*15,dim=0)
        right_lung_6x6_2 = torch.Tensor(np.array([[1,1,1,1,1,1],
                                                [1,1,1,1,1,0],
                                                [1,1,1,0,0,0],
                                                [1,1,1,0,0,0],
                                                [1,0,0,0,0,0],
                                                [0,0,0,0,0,0]]))
        right_lung_seg_small2 = torch.stack([right_lung_6x6_2]*15,dim=0)
        left_lung_6x6_2 = torch.Tensor(np.array([[0,1,1,1,1,1],
                                               [0,0,0,0,1,1],
                                               [0,0,0,0,1,1],
                                               [0,0,0,0,0,0],
                                               [0,0,0,1,1,1],
                                               [0,0,0,1,1,1]]))
        left_lung_seg_small2 = torch.stack([left_lung_6x6_2]*15,dim=0)
        both_lung_6x6_2 = torch.Tensor(np.array([[1,1,1,1,1,1],
                                                [1,1,1,1,1,1],
                                                [1,1,1,0,1,1],
                                                [1,1,1,0,0,0],
                                                [1,0,0,1,1,1],
                                                [0,0,0,1,1,1]]))
        both_lung_seg_small2 = torch.stack([both_lung_6x6_2]*15,dim=0)
        sample2 = {}
        sample2['heart_gr_truth'] = np.array([1,1,1])
        sample2['left_lung_gr_truth'] = np.array([1,0,1,1,0])
        sample2['right_lung_gr_truth'] = np.array([0,0,1,1,1])
        attn_gr_truth_out2 = mask.construct_attn_gr_truth(sample2, heart_seg_small2, left_lung_seg_small2, right_lung_seg_small2)
        attn_gr_truth_correct2 = torch.stack([heart_seg_small2,heart_seg_small2,heart_seg_small2,
                                              left_lung_seg_small2,zeros_seg_small,
                                              both_lung_seg_small2,both_lung_seg_small2,
                                              right_lung_seg_small2],dim=0)
        for disease_idx in range(8):
            assert (attn_gr_truth_out2[disease_idx,:,:,:]==attn_gr_truth_correct2[disease_idx,:,:,:]).all()
        assert (attn_gr_truth_out2==attn_gr_truth_correct2).all()
        print('Passed test_construct_attn_gr_truth()')

if __name__=='__main__':
    unittest.main()