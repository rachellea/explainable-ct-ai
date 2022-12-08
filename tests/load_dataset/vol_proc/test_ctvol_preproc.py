#test_ctvol_preproc.py
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

import copy
import torch
import unittest
import numpy as np
import pandas as pd

from src.load_dataset.vol_proc import ctvol_preproc
from tests import equality_checks as eqc

array3d = np.array([[[1,2],[3,4]],
        [[-1,-2],[-3,-4]],
        [[6,2],[1,9]],
        [[5,3],[1,1]],
        [[8,8],[4,7]]])  #shape (5, 2, 2)

class TestCTVolumePreprocessing(unittest.TestCase):
    def test_pad_slices_and_sides(self):
        global array3d
        out = ctvol_preproc.pad_slices(array3d, 6)
        cor = np.array([[[1,2],[3,4]],
        [[-1,-2],[-3,-4]],
        [[6,2],[1,9]],
        [[5,3],[1,1]],
        [[8,8],[4,7]],
        [[-4,-4],[-4,-4]]])
        assert eqc.arrays_equal(out, cor)
        
        z = copy.deepcopy(array3d)
        z[0,0,0] = -10
        out2 =  ctvol_preproc.pad_sides(z, 3)
        cor2 = np.array([[[-10,2,-10],[3,4,-10],[-10,-10,-10]],
        [[-1,-2,-10],[-3,-4,-10],[-10,-10,-10]],
        [[6,2,-10],[1,9,-10],[-10,-10,-10]],
        [[5,3,-10],[1,1,-10],[-10,-10,-10]],
        [[8,8,-10],[4,7,-10],[-10,-10,-10]]])
        assert eqc.arrays_equal(out2, cor2)
        print('Passed test_pad_slices_and_sides()')
    
    def test_sliceify(self):
        global array3d
        out = ctvol_preproc.sliceify(ctvol_preproc.pad_slices(array3d, 6))
        cor = np.array([[[[1,2],[3,4]],
        [[-1,-2],[-3,-4]],
        [[6,2],[1,9]]],
    
        [[[5,3],[1,1]],
        [[8,8],[4,7]],
        [[-4,-4],[-4,-4]]]])
        assert eqc.arrays_equal(out, cor)
        assert cor.shape == (2, 3, 2, 2)
        print('Passed test_sliceify()')
    
    def test_normalize(self):
        lower_bound = -1000
        upper_bound = 200
        test = np.reshape(np.array([500,3000,-150,-1000,-1500,130]), (6,1,1))
        out = ctvol_preproc.normalize(torch.Tensor(test), lower_bound = lower_bound, upper_bound = upper_bound)
        cor = np.reshape(np.array([1,1,0.70833333,0,0,0.941666666]), (6,1,1))
        assert eqc.arrays_equal(out.numpy(), cor)
        print('Passed test_normalize()')
    
    def test_crop_specified_axis(self):
        ctvol = np.array([[[1],[2],[3]],
                          [[4],[5],[6]],
                          [[7],[8],[9]]])
        out = ctvol_preproc.crop_specified_axis(copy.deepcopy(ctvol), max_dim=2, axis=0)
        cor = np.array([[[1],[2],[3]],
                        [[4],[5],[6]]])
        assert eqc.arrays_equal(out, cor)
        out2 = ctvol_preproc.crop_specified_axis(copy.deepcopy(ctvol), max_dim=1, axis=1)
        cor2 = np.array([[[2]],[[5]],[[8]]])
        assert eqc.arrays_equal(out2,cor2)
        
        ctvol = np.array([[  [0,1,2], [3,4,5]  ],
                          [  [6,7,8], [9,10,11]  ],
                          [  [12,13,14], [15,16,17]  ]])
        out3 = ctvol_preproc.crop_specified_axis(copy.deepcopy(ctvol), max_dim=2, axis=2)
        cor3 = np.array([[  [0,1], [3,4]  ],
                          [  [6,7], [9,10]  ],
                          [  [12,13], [15,16]  ]])
        assert eqc.arrays_equal(out3,cor3)
        print('Passed test_crop_specified_axis()')
    
    def test_adjust_coordinates(self):
        #mino1 would go below 0
        mino1, maxo1 = ctvol_preproc.adjust_coordinates(5,10,20,22)
        assert mino1 == 0 #instead of -2
        assert maxo1 == 10+8+2
        
        #maxo2 would go above axis_length
        mino2, maxo2 = ctvol_preproc.adjust_coordinates(300,336,102,337)
        assert mino2 == 235
        assert maxo2 == 337
        
        #both mino3 would go below 0 and max03 would go above axis_length
        mino3, maxo3 = ctvol_preproc.adjust_coordinates(2,4,800,5)
        assert mino3 == 0
        assert maxo3 == 5
        
        #mino4 and maxo4 would be fine
        mino4, maxo4 = ctvol_preproc.adjust_coordinates(400,450,60,500)
        assert mino4 == 395
        assert maxo4 == 455
        
        #mino5 and maxo5 are already the right size and don't change
        mino5, maxo5 = ctvol_preproc.adjust_coordinates(100,200,100,300)
        assert mino5==100
        assert maxo5==200
        
        #mino6 and maxo6 are too big and need to be brought closer together
        mino6, maxo6 = ctvol_preproc.adjust_coordinates(100,200,50,300)
        assert mino6==125
        assert maxo6==175
        
        #mino7 and maxo7 are too big and need to be brought closer together
        mino7, maxo7 = ctvol_preproc.adjust_coordinates(100,150,20,300)
        assert mino7==115
        assert maxo7==135
        
        print('Passed test_adjust_coordinates()')
    
    def test_fix_extrema(self):
        data = np.array([np.nan,np.inf,-np.inf,-np.inf,np.inf,np.nan,100,120,113])
        data = np.expand_dims(data, 0) #so it has shape (9,1) instead of (9,)
        columns = ['sup_axis0min','inf_axis0max','ant_axis1min','pos_axis1max',
                   'rig_axis2min','lef_axis2max','shape0','shape1','shape2']
        extrema = pd.DataFrame(data,columns=columns,index=['AA12345'])
        out = ctvol_preproc.fix_extrema(extrema)
        correctdata = np.array([0,100,0,120,0,113,100,120,113])
        correctdata = np.expand_dims(correctdata, 0)
        correct = pd.DataFrame(correctdata,columns=columns,index=['AA12345'])
        assert (out == correct).all().all()
        print('Passed test_fix_extrema()')
    
    def test_fix_right_border(self):
        right_values = [20,100,136,200]
        right_answers = [20,100,318,350] #mean (136+500)/2 = 318 etc.
        for idx in range(len(right_values)):
            right_value = right_values[idx]
            right_answer = right_answers[idx]
            columns = ['sup_axis0min','inf_axis0max','ant_axis1min','pos_axis1max',
                       'rig_axis2min','lef_axis2max','shape0','shape1','shape2']
            extremadata = np.array([[30,340,106,347, right_value, 390,512,512,512],
                                    [30,340,106,347,500,390,512,512,512]])
            extrema = pd.DataFrame(extremadata,columns=columns,index=['AA12345','AAzzz'])
            extrema['Subset_Assigned']='imgtrain'
            rig_axis2min = ctvol_preproc.fix_right_border(extrema.at['AA12345','rig_axis2min'], extrema)
            assert rig_axis2min == right_answer
        print('Passed test_fix_right_border()')
        
    def test_fix_left_border(self):
        left_values = [100,290,400,420]
        left_answers = [300,395,400,420] #mean (100+500)/2 = 300 etc.
        for idx in range(len(left_values)):
            left_value = left_values[idx]
            left_answer = left_answers[idx]
            columns = ['sup_axis0min','inf_axis0max','ant_axis1min','pos_axis1max',
                       'rig_axis2min','lef_axis2max','shape0','shape1','shape2']
            extremadata = np.array([[30,340,106,347,65, left_value, 512,512,512],
                                   [30,340,106,347,65,500,512,512,512]])
            extrema = pd.DataFrame(extremadata,columns=columns,index=['AA12345','AAzzz'])
            extrema['Subset_Assigned']='imgtrain'
            lef_axis2max = ctvol_preproc.fix_left_border(extrema.at['AA12345','lef_axis2max'], extrema)
            assert lef_axis2max == left_answer
        print('Passed test_fix_left_border()')
    
    def test_placeholders(self):
        """Check that placeholders work correctly.
        Documentation of resize_:
        'Resizes self tensor to the specified size. If the number of elements is
        larger than the current storage size, then the underlying storage is resized
        to fit the new number of elements. If the number of elements is smaller,
        the underlying storage is not changed. Existing elements are preserved
        but any new memory is uninitialized. WARNING: This is a low-level method.
        The storage is reinterpreted as C-contiguous, ignoring the current strides
        (unless the target size equals the current size, in which case the tensor
        is left unchanged). For most purposes, you will instead want to use view(),
        which checks for contiguity, or reshape(), which copies data if needed.
        To change the size in-place with custom strides, see set_().'
        
        Documentation of copy_:
        'copy_(src, non_blocking=False) -> Tensor
        Copies the elements from src into self tensor and returns self.
        The src tensor must be broadcastable with the self tensor. It may be of a
        different data type or reside on a different device.'
        """
        device = torch.device('cuda:0')
        placeholder_data = torch.FloatTensor(1).to(device)
        placeholder_gr_truth = torch.FloatTensor(1).to(device)
        
        #Make sure that placeholders accurately reflect the new data added to them
        data1 = torch.Tensor([1,0.9,3.6,44,32.1]).to(device)
        gr_truth1 = torch.Tensor([98,-8.54,23.7,-3]).to(device)
        placeholder_data.resize_(data1.shape).copy_(data1)
        placeholder_gr_truth.resize_(gr_truth1.shape).copy_(gr_truth1)
        assert (placeholder_data==torch.Tensor([1,0.9,3.6,44,32.1]).to(device)).all()
        assert (placeholder_gr_truth==torch.Tensor([98,-8.54,23.7,-3]).to(device)).all()
        
        data2 = torch.Tensor([[-1,2],[4,1.234]]).to(device)
        gr_truth2 = torch.Tensor([[[0.9,0.1],[-0.34,1.1]],[[9,8],[7,6]]]).to(device)
        placeholder_data.resize_(data2.shape).copy_(data2)
        placeholder_gr_truth.resize_(gr_truth2.shape).copy_(gr_truth2)
        assert (placeholder_data==torch.Tensor([[-1,2],[4,1.234]]).to(device)).all()
        assert (placeholder_gr_truth==torch.Tensor([[[0.9,0.1],[-0.34,1.1]],[[9,8],[7,6]]]).to(device)).all()
        
        data3 = torch.rand(23,54,63,24).to(device).float()
        gr_truth3 = torch.rand(1,24,31).to(device).float()
        placeholder_data.resize_(data3.shape).copy_(data3)
        placeholder_gr_truth.resize_(gr_truth3.shape).copy_(gr_truth3)
        assert (placeholder_data==copy.deepcopy(data3)).all()
        assert (placeholder_gr_truth==copy.deepcopy(gr_truth3)).all()
        print('Done with test_placeholders()')

if __name__ == '__main__':
    unittest.main()