#custom_models_triple_orig.py
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
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from torchvision import transforms, models, utils

#Set seeds
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

class ResNet18_TripleCrop(nn.Module):
    """Model for big data. ResNet18 then 3D conv then FC.
    Here the input x is a Python dictionary with keys right_lung, heart, and
    left_lung.
    This model applies the same ResNet and 3D conv feature extractor to each crop,
    then applies separate FC layers to make the final predictions.
    Note that running this model requires model parallelism because it does not
    fit on a single GPU."""
    def __init__(self, n_outputs_lung, n_outputs_heart):
        super(ResNet18_TripleCrop, self).__init__()
        
        #Note that x = x.to('cuda:0') means x will be sent to the first GPU
        #that torch can see. So if you've done CUDA_VISIBLE_DEVICES=2,3 then
        #'cuda:0' may correspond to GPU 2 in reality.
        resnet = models.resnet18(pretrained=True)
        ##in total the resnet has 10 children. You are going to use 0:-2 of them
        self.features1 = nn.Sequential(*(list(resnet.children())[0:4])).to('cuda:1')
        self.features2 = nn.Sequential(*(list(resnet.children())[4:-2])).to('cuda:0')
        
        #conv input torch.Size[1,134,512,10,5]
        self.reducingconvs = nn.Sequential(
            nn.Conv3d(134, 64, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(64, 32, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(32, 16, kernel_size = (3,2,2), stride=(3,2,2), padding=0),
            nn.ReLU(inplace=True)).to('cuda:0')
        
        self.classifier_heart = nn.Sequential(
            nn.Linear(16*18*5*1, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(128, 96), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(96, n_outputs_heart)).to('cuda:0')
        
        self.classifier_lung = nn.Sequential(
            nn.Linear(16*18*5*1, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(128, 96), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(96, n_outputs_lung)).to('cuda:0')
            
    def extract_organ_representation(self, z):
        """z is a crop from the full volume representing the right_lung, heart,
        or left_lung"""
        #z input shape is [1, 134, 3, 420, 210]
        z = z.to('cuda:1')
        shape = list(z.size())
        batch_size = int(shape[0])
        z = z.view(batch_size*134,3,420,210)
        z = self.features1(z)
        z = z.to('cuda:0')
        z = self.features2(z) #after features: [134, 512, 14, 7]
        z = z.unsqueeze(0) #after unsqueeze: [1, 134, 512, 14, 7]
        z = self.reducingconvs(z) #after convs: [1, 16, 18, 5, 1]
        z = z.view(1, 16*18*5*1)
        return z
    
    def forward(self, x):
        #Extract features using the same ResNet and 3D conv layers:
        #Note that x is a dictionary with keys right_lung, heart, and left_lung
        right_lung = self.extract_organ_representation(x['right_lung'])
        heart = self.extract_organ_representation(x['heart'])
        left_lung = self.extract_organ_representation(x['left_lung'])
        
        #Perform classification that is specific to the organ
        #Lungs share the classifier. Heart has a separate classifier.
        right_lung = self.classifier_lung(right_lung)
        heart = self.classifier_heart(heart)
        left_lung = self.classifier_lung(left_lung)
        
        #Now concatenate to get the final label vector
        #Order: heart, left_lung, right_lung (lexicographic because that is the
        #order of the ground truth)
        return torch.cat((heart,left_lung,right_lung),1)


class ResNet18_TripleCrop_Seg(nn.Module):
    """Same as ResNet18_TripleCrop except assumes input CT size is smaller"""
    def __init__(self, n_outputs_lung, n_outputs_heart):
        super(ResNet18_TripleCrop_Seg, self).__init__()        
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*(list(resnet.children())[:-2]))
        
        #conv input torch.Size[1,134,512,10,5]
        self.reducingconvs = nn.Sequential(
            nn.Conv3d(107, 64, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(64, 32, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(32, 16, kernel_size = (3,2,2), stride=(3,2,2), padding=0),
            nn.ReLU(inplace=True))
        
        self.classifier_heart = nn.Sequential(
            nn.Linear(16*18*3*1, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(128, 96), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(96, n_outputs_heart))
        
        self.classifier_lung = nn.Sequential(
            nn.Linear(16*18*3*1, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(128, 96), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(96, n_outputs_lung))
            
    def extract_organ_representation(self, z):
        shape = list(z.size()) #e.g. [1, 107, 3, 340, 170]
        batch_size = int(shape[0])
        z = z.view(batch_size*107,3,340,170)
        z = self.features(z) #after features: [107, 512, 11, 6]
        z = z.unsqueeze(0) #after unsqueeze: [1, 107, 512, 11, 6]
        z = self.reducingconvs(z) #after convs: [1, 16, 18, 3, 1]
        z = z.view(1, 16*18*3*1)
        return z
    
    def forward(self, x):
        right_lung = self.extract_organ_representation(x['right_lung'])
        heart = self.extract_organ_representation(x['heart'])
        left_lung = self.extract_organ_representation(x['left_lung'])
        right_lung = self.classifier_lung(right_lung)
        heart = self.classifier_heart(heart)
        left_lung = self.classifier_lung(left_lung)
        return torch.cat((heart,left_lung,right_lung),1)


class ResNet18_TripleCrop_Seg_OneOrgan(nn.Module):
    """Works for one organ at a time e.g. heart only, left lung only, or right
    lung only. Assumes organ crop comes from a segmentation-based CT center
    crop (smaller organ crop)"""
    def __init__(self, n_outputs_organ, organ_name):
        super(ResNet18_TripleCrop_Seg_OneOrgan, self).__init__()  
        
        self.organ_name = organ_name
         
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*(list(resnet.children())[:-2]))
        
        #conv input torch.Size[1,134,512,10,5]
        self.reducingconvs = nn.Sequential(
            nn.Conv3d(107, 64, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(64, 32, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(32, 16, kernel_size = (3,2,2), stride=(3,2,2), padding=0),
            nn.ReLU(inplace=True))
        
        self.classifier_organ = nn.Sequential(
            nn.Linear(16*18*3*1, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(128, 96), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(96, n_outputs_organ))
            
    def extract_organ_representation(self, z):
        shape = list(z.size()) #e.g. [1, 107, 3, 340, 170]
        batch_size = int(shape[0])
        z = z.view(batch_size*107,3,340,170)
        z = self.features(z) #after features: [107, 512, 11, 6]
        z = z.unsqueeze(0) #after unsqueeze: [1, 107, 512, 11, 6]
        z = self.reducingconvs(z) #after convs: [1, 16, 18, 3, 1]
        z = z.view(1, 16*18*3*1)
        return z
    
    def forward(self, x):
        organ = self.extract_organ_representation(x[self.organ_name])
        return self.classifier_organ(organ)


class ResNet18_TripleCrop_NoSeg_OneOrgan(nn.Module):
    """Works for one organ at a time e.g. heart only, left lung only, or right
    lung only. Assumes organ crop comes from a generic CT center crop
    (larger organ crop)"""
    def __init__(self, n_outputs_organ, organ_name):
        super(ResNet18_TripleCrop_NoSeg_OneOrgan, self).__init__()
        
        self.organ_name = organ_name
         
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*(list(resnet.children())[:-2]))
                
        #conv input torch.Size[1,134,512,10,5]
        self.reducingconvs = nn.Sequential(
            nn.Conv3d(134, 64, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(64, 32, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(32, 16, kernel_size = (3,2,2), stride=(3,2,2), padding=0),
            nn.ReLU(inplace=True)).to('cuda:0')
        
        self.classifier_organ = nn.Sequential(
            nn.Linear(16*18*5*1, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(128, 96), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(96, n_outputs_organ))
            
    def extract_organ_representation(self, z):
        shape = list(z.size())
        batch_size = int(shape[0])
        z = z.view(batch_size*134,3,420,210)
        z = self.features(z)
        z = z.unsqueeze(0)
        z = self.reducingconvs(z)
        z = z.view(1, 16*18*5*1)
        return z
    
    def forward(self, x):
        organ = self.extract_organ_representation(x[self.organ_name])
        return self.classifier_organ(organ)