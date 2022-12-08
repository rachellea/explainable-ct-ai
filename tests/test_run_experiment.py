#test_run_experiment.py
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

import unittest
import numpy as np

from tests import equality_checks as eqc

class TestNumpyBatchTracking(unittest.TestCase):
    def test_numpy_gr_truth_and_perf_tracking(self):
        """This replicates some of the code in run_experiment.py that tracks
        predicted probabilities and ground truth across batches, to make sure
        that it works"""
        def return_fake_preds(batch_idx):
            if batch_idx==0: #3 examples
                pred_np = np.array([[0.16,0.22,0.36,0.46,0.15],
                                    [0.56,0.23,0.83,0.86,0.27],
                                    [0.27,0.05,0.03,0.45,0.23]])
                gr_truth_np = np.array([[1,0,0,1,0],
                                        [0,1,0,0,0],
                                        [1,1,1,1,1]])
                volume_accs_np = np.array(['1111.npz','2222.npz','3333.npz'])
            elif batch_idx==1: #3 examples
                pred_np = np.array([[0.93,0.45,0.37,0.13,0.67],
                                    [0.34,0.63,0.34,0.34,0.23],
                                    [0.65,0.54,0.43,0.32,0.21]])
                gr_truth_np = np.array([[1,0,0,0,1],
                                        [0,0,0,0,0],
                                        [0,1,1,1,0]])
                volume_accs_np = np.array(['1029.npz','3535.npz','9586.npz'])
            elif batch_idx==2: #1 example
                pred_np = np.array([[0.26,0.37,0.48,0.59,0.60]])
                gr_truth_np = np.array([[1,1,0,0,1]])
                volume_accs_np = np.array(['9876.npz'])
            return pred_np, gr_truth_np, volume_accs_np
                
        
        #Initialize numpy arrays for storing results. examples x labels
        #Do NOT use concatenation, or else you will have memory fragmentation.
        num_examples = 7
        num_labels = 5
        batch_size = 3
        pred_epoch = np.zeros([num_examples,num_labels])
        gr_truth_epoch = np.zeros([num_examples,num_labels])
        volume_accs_epoch = np.empty(num_examples,dtype='U32') #need to use U32 to allow string of length 32
        
        for batch_idx in range(3):
            #Save predictions and ground truth across batches
            pred_np, gr_truth_np, volume_accs_np = return_fake_preds(batch_idx)
            start_row = batch_idx*batch_size
            stop_row = min(start_row + batch_size, num_examples)
            pred_epoch[start_row:stop_row,:] = pred_np #pred_epoch is e.g. [25355,80] and pred_np is e.g. [1,80] for a batch size of 1
            gr_truth_epoch[start_row:stop_row,:] = gr_truth_np #gr_truth_epoch has same shape as pred_epoch
            volume_accs_epoch[start_row:stop_row] = volume_accs_np #volume_accs_epoch stores the volume accessions in the order they were used
        
        #Correct answers
        correct_pred_epoch = np.array([[0.16,0.22,0.36,0.46,0.15],
                                    [0.56,0.23,0.83,0.86,0.27],
                                    [0.27,0.05,0.03,0.45,0.23],
                                    [0.93,0.45,0.37,0.13,0.67],
                                    [0.34,0.63,0.34,0.34,0.23],
                                    [0.65,0.54,0.43,0.32,0.21],
                                    [0.26,0.37,0.48,0.59,0.60]])
        correct_gr_truth_epoch =  np.array([[1,0,0,1,0],
                                        [0,1,0,0,0],
                                        [1,1,1,1,1],
                                        [1,0,0,0,1],
                                        [0,0,0,0,0],
                                        [0,1,1,1,0],
                                        [1,1,0,0,1]])
        correct_volume_accs_epoch = np.array(['1111.npz','2222.npz','3333.npz','1029.npz','3535.npz','9586.npz','9876.npz'])
        
        assert eqc.arrays_equal(pred_epoch, correct_pred_epoch)
        assert eqc.arrays_equal(gr_truth_epoch, correct_gr_truth_epoch)
        assert (volume_accs_epoch==correct_volume_accs_epoch).all()
        print('Passed test_numpy_gr_truth_and_perf_tracking()')

if __name__=='__main__':
    unittest.main()