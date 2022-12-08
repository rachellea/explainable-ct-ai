#custom_models_text.py
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

import numpy as np
import torch, torch.nn as nn

#Set seeds
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

class TextClassifier1(nn.Module):
    """2D CNN to classify whether text is present in each slice of the CT
    scan"""
    def __init__(self, slices): #TODO - make it so you don't need 'slices' as a dummy arg to satisfy run_experiment.py
        super(TextClassifier1, self).__init__()
        self.conv2d = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size = (3,3), stride=(3,3), padding=0),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(32, 32, kernel_size = (3,3), stride=(3,3), padding=0),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(32, 64, kernel_size = (3,3), stride=(3,3), padding=0),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(64, 64, kernel_size = (3,3), stride=(3,3), padding=0),
                nn.ReLU(inplace=True))
        self.fc = nn.Linear(1600, 1)
        
    def forward(self, x):
        assert list(x.shape)==[1,405,420,420]
        x = x.squeeze() #out shape [405,420,420]
        x = x.unsqueeze(dim=1) #out shape [405, 1, 420, 420], one channel
        x = self.conv2d(x) #out shape [405, 64, 5, 5]
        x = x.flatten(start_dim=1, end_dim=-1) #out shape [405, 1600]
        x = self.fc(x) #out shape [405, 1]
        #now transpose, since the ground truth has 405 labels in it and we're
        #pretending that this is a batch size of 1 (so we can make use of
        #existing CT scan preprocessing code)
        x = x.transpose(0,1) #out shape [1, 405]
        return x

class TextClassifier2(nn.Module):
    """2D CNN to classify whether text is present in each slice of the CT
    scan. Uses adaptive max pooling so that it can be applied to inputs
    of various shapes."""
    def __init__(self, slices): #TODO - make it so you don't need 'slices' as a dummy arg to satisfy run_experiment.py
        super(TextClassifier2, self).__init__()
        self.conv2d = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size = (3,3), stride=(3,3), padding=0),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(32, 32, kernel_size = (3,3), stride=(3,3), padding=0),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(32, 64, kernel_size = (3,3), stride=(3,3), padding=0),
                nn.ReLU(inplace=True))
        
        #adaptive max pool everything to shape (5,5)
        #use adaptive max pooling instead of adaptive avg pooling because if
        #deploying a model on data padded to the max size, most data will have
        #a lot of black padding (no text) which could unfairly 'decrease'
        #important features if adaptive avg pooling were used. 
        self.adaptivemax = nn.AdaptiveMaxPool2d((5,5)) 
        
        self.fc = nn.Linear(1600, 1)
        
    def forward(self, x):
        x = x.squeeze() #out shape [405,405,405]
        x = x.unsqueeze(dim=1) #out shape [405, 1, 405, 405], one channel
        x = self.conv2d(x) #out shape for 'cube' data is [405, 64, 15, 15]
        x = self.adaptivemax(x) #out shape [405, 64, 5, 5]
        x = x.flatten(start_dim=1, end_dim=-1) #out shape [405, 1600]
        x = self.fc(x) #out shape [405, 1]
        #now transpose, since the ground truth has 405 labels in it and we're
        #pretending that this is a batch size of 1 (so we can make use of
        #existing CT scan preprocessing code)
        x = x.transpose(0,1) #out shape [1, 405]
        return x
