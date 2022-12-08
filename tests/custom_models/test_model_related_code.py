#test_model_related_code.py
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

import unittest
import numpy as np

import torch
import torch.nn as nn

from tests import equality_checks as eqc

class TestModelRelatedCode(unittest.TestCase):
    def test_reshape_and_view(self):
        """Ensure that reshaping and viewing using numpy and torch has the desired
        effects. Needed in order to use pretrained 2D feature extractor on a 3D
        network"""
        device = torch.device('cuda:0')
        x_orig = np.random.rand(6,140,3,50,50) #batch size, height, channels, squareside, squareside
        x = torch.from_numpy(x_orig).squeeze().type(torch.float)
        
        #does torch view reverse numpy reshape?:
        s = x.shape
        x = np.reshape(x, (s[0]*s[1],s[2],s[3],s[4]))
        x = x.to(device)
        x = x.view(6,140,3,50,50)
        xout1 = x.cpu().numpy()
        assert eqc.arrays_equal(xout1, x_orig)
        
        #Does torch reverse itself?
        x = x.view(6,140*3*50*50)
        x = x.view(6,140,3,50,50)
        xout2 = x.cpu().numpy()
        assert eqc.arrays_equal(xout2, x_orig)
        print('Passed test_reshape_and_view()')
    
    def test_reshape_and_view_resnet18_batch(self):
        #batch size 2
        x = torch.Tensor(np.random.rand(2,13,3,42,42))
        shape = list(x.size())
        batch_size = int(shape[0])
        y = x.view(batch_size*13, 3, 42, 42)
        y = y.view(batch_size, 13, 3, 42, 42)
        assert int((x == y).all())==1
        z = x.view(batch_size,13*3*42*42)
        z = z.view(batch_size,13,3,42,42)
        assert int((z == x).all())==1
        #batch size 1
        x = torch.Tensor(np.random.rand(1,13,3,42,42))
        shape = list(x.size())
        batch_size = int(shape[0])
        y = x.view(batch_size*13,3,42,42)
        y = y.view(batch_size, 13, 3, 42, 42)
        assert int((x==y).all())==1
        z = x.view(batch_size,13*3*42*42)
        z = z.view(batch_size,13,3,42,42)
        assert int((z==x).all())==1
        print('Passed test_reshape_and_view_resnet18_batch()')
    
    def test_reshape_and_view_bodyconv(self):
        #Basic test
        newx = torch.Tensor(np.random.rand(13,51,14,14))
        newx2 = newx.transpose(0,1).unsqueeze(0)
        newx2 = newx2.squeeze(0).transpose(0,1)
        assert int((newx==newx2).all())==1
        
        #Complex test (see paper hand drawing in C:\Users\Rachel\Documents\CarinLab\Project_Radiology\2020-01_January\2020-01-16-David-Code-BodyConvFix-My-Verification
        #for visual verification)
        mini = torch.Tensor(np.array([[[[5, 8],[3, 7]],
             [[2, 2],[8, 3]],
             [[1, 9],[4, 5]],
             [[3, 9],[7, 9]],
             [[3, 2],[6, 6]]],
    
            [[[5, 8],[4, 8]],
             [[7, 7],[8, 6]],
             [[6, 1],[9, 9]],
             [[4, 9],[6, 6]],
             [[2, 2],[6, 4]]],
    
            [[[4, 8],[2, 5]],
             [[5, 7],[9, 8]],
             [[1, 9],[4, 4]],
             [[2, 8],[1, 3]],
             [[8, 8],[1, 3]]]]))
        output = mini.transpose(0,1).unsqueeze(0)
        correct = torch.Tensor(np.array([[[[[5, 8],[3, 7]],
              [[5, 8],[4, 8]],
              [[4, 8],[2, 5]]],
    
             [[[2, 2],[8, 3]],
              [[7, 7],[8, 6]],
              [[5, 7],[9, 8]]],
    
             [[[1, 9],[4, 5]],
              [[6, 1],[9, 9]],
              [[1, 9],[4, 4]]],
    
             [[[3, 9],[7, 9]],
              [[4, 9],[6, 6]],
              [[2, 8],[1, 3]]],
    
             [[[3, 2],[6, 6]],
              [[2, 2],[6, 4]],
              [[8, 8],[1, 3]]]]]))
        assert int((output==correct).all())==1
        print('Passed test_reshape_and_view_bodyconv()')
    
    def test_reshape_and_view_pool(self):
        #The right way to think about it is the effective kernel size across
        #the 512 dimension, which is in fact 3*3*3=27 (because the kernel size
        #is 3 in the 512 direction and its stride in that direction is 3.) 
        #Don't get distracted by reducingpools2 which are across the 134 dimension.
        #So, we actually have (for the 512 direction):
        #0-27, 27-54, 54-81, 81-108, 108-135, 135-162, 162-189, 189-216, 216-243,
        #243-270, 270-297, 297-324, 324-351, 351-378, 378-405, 405-432, 432-459,
        #459-486, 486-513
        x = np.zeros([1,134,512,14,14],dtype='int')
        x[:,:,14,:,:] = 0; x[:,:,42,:,:] = 1
        x[:,:,70,:,:] = 2; x[:,:,98,:,:] = 3
        x[:,:,126,:,:] = 4; x[:,:,154,:,:] = 5
        x[:,:,182,:,:] = 6; x[:,:,210,:,:] = 7
        x[:,:,238,:,:] = 8; x[:,:,266,:,:] = 9
        x[:,:,294,:,:] = 10; x[:,:,322,:,:] = 11
        x[:,:,348,:,:] = 12; x[:,:,372,:,:] = 13
        x[:,:,400,:,:] = 14; x[:,:,420,:,:] = 15
        x[:,:,440,:,:] = 16; x[:,:,465,:,:] = 17
        #27*18 = 486. If we put any value into 486 or higher, it won't be found in
        #the final result, which honestly disturbs me.
        #But if we put anything huge into 485 or lower, it will show up in the 17 slot
        #in the final output. 
        x[:,:,486,:,:] = 99999 #doesn't show up in final output which is disturbing
        x = torch.Tensor(x)
        
        reducingpools = nn.Sequential(
            nn.MaxPool3d(kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(),
        
            nn.MaxPool3d(kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(),
        
            nn.MaxPool3d(kernel_size = (3,2,2), stride=(3,2,2), padding=0),
            nn.ReLU())
            
        reducingpools2 = nn.Sequential(
            nn.MaxPool3d(kernel_size = (8,1,1), stride=(8,1,1), padding=0),
            nn.ReLU())
        
        shape = list(x.size())
        batch_size = int(shape[0])
        x = reducingpools(x)
        assert batch_size == 1
        x = torch.squeeze(x) #size [134, 18, 5, 5]
        x = x.transpose(0,1) #size [18, 134, 5, 5]
        x = reducingpools2(x) #Output is [18, 16, 5, 5]
        x = x.transpose(0,1) #size [16, 18, 5, 5]
        x = x.unsqueeze(0) #size [1, 16, 18, 5, 5]
        x = x.contiguous().numpy()
        
        for number in range(0,18):
            print(number)
            selected = list(set(x[:,:,number,:,:].flatten().tolist()))
            print(selected)
            assert len(selected)==1, 'len='+str(len(selected))
            assert int(selected[0]) == (number), 'int(selected[0])='+str(int(selected[0]))+' number+1='+str(number)
        print('Passed test_reshape_and_view_pool()')

if __name__=='__main__':
    unittest.main()