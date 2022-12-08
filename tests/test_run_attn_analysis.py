#test_run_attention_analysis.py
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

import os
import shutil
import unittest
import numpy as np
import pandas as pd

from tests import equality_checks as eqc
from src import run_attn_analysis

class TestMakeLabelIndicesDict(unittest.TestCase):
    def test_make_label_indices_dict(self):
        """Test the code that splits up labels for a particular scan according
        to whether the predictions are true positives, true negatives, false
        positives, or false negatives."""
        #Create fake paths
        if not os.path.exists('temporary-test-dir'):
            os.mkdir('temporary-test-dir')
        if not os.path.exists('temporary-test-dir/pred_probs'):
            os.mkdir('temporary-test-dir/pred_probs')
        params_path = 'temporary-test-dir/params/fakemodelname'
        
        #Create fake data
        volume_acc = '12345.npz'
        gr_truth = np.array([0,1,0,1,0,1])
        label_meanings = ['abn1','abn2','abn3','abn4','abn5','abn6']
        
        #We're focused on the middle row, for the volume acc 12345.npz
        #The binarized predictions (based on medians) are [1,0,0,1,1]
        #When compared with the ground truth this conceptually yields:
        #[false positive, false negative, true negative, true positive,
        #false positive, true positive]
        pred_probs_all = pd.DataFrame([[0.1, 0.99, 0.81, 0.3, 0.5, 0.010],
                                       [0.5, 0.90, 0.40, 0.3, 0.6, 0.030], #relevant one
                                       [0.2, 0.98, 0.44, 0.3, 0.9, 0.002]],
                index=['999.npz','12345.npz','222.npz'], columns=label_meanings)
        pred_probs_all.to_csv('./temporary-test-dir/pred_probs/valid_predprob_ep74.csv',header=True,index=True)
        
        gr_truth_all =   pd.DataFrame([[1,1,1,0,0,0],
                                       [0,1,0,1,0,1], #relevant one
                                       [0,0,1,1,0,0]],
                index=['999.npz','12345.npz','222.npz'], columns=label_meanings)
        gr_truth_all.to_csv('./temporary-test-dir/pred_probs/valid_grtruth_ep74.csv',header=True,index=True)
        
        #Run
        output_label_indices_dict = run_attn_analysis.make_label_indices_dict(volume_acc, volume_acc, gr_truth, params_path, label_meanings)
        
        #Check
        correct_label_indices_dict =  {'g0p0':[2], #true negatives
                          'g0p1':[0,4], #false positives
                          'g1p0':[1], #false negatives
                          'g1p1':[3,5]} #true positives
        for key in (list(output_label_indices_dict.keys())+list(correct_label_indices_dict.keys())):
            assert output_label_indices_dict[key]==correct_label_indices_dict[key]
        
        #Clean up
        shutil.rmtree('./temporary-test-dir')
        print('Passed test_make_label_indices_dict()')

if __name__=='__main__':
    unittest.main()