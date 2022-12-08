#custom_models_multiview.py
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

#Import components
from . import components as cts

class BodyAvgMultiview(nn.Module):
    """(1) ResNet18 [slices, 512, 14, 14]
       (2) shared conv_final to [slices, 16, 8, 1]
       (3) FC layers (implemented via conv) to [n_outputs, 1, 1]
           There is a separate FC layer for each view: axial, coronal, sagittal
       (4) Avg, max, or learned weighted avg pooling over slices to
           get [n_outputs]
    
    Variables:
    <chosen_conv>: either 'sharedconv' to share the 2d conv across all views
        or 'separateconv' to have separate 2d conv for each view
    <chosen_comb_func>: str, any of:
        'avg' to avg the disease predictions of each of the views
        'max' to take the max of the disease predictions across the views
        'weighted_avg' to take the weighted average of the disease
            predictions across the views where each disease is associated with
            3 learned coefficients that weight the views. A softmax is applied
            per disease so that the sum of the coefficients is one for each
            disease."""
    def __init__(self, n_outputs, chosen_conv, chosen_comb_func):
        super(BodyAvgMultiview, self).__init__()
        self.slices = 15 #9 projections
        self.chosen_comb_func = chosen_comb_func
        self.features = cts.resnet_features()
        
        #Set up convolution
        print('Using',chosen_conv,'for multiple views')
        assert chosen_conv in ['sharedconv','separateconv']
        if chosen_conv == 'sharedconv':
            sharedconv = cts.final_conv()
            self.conv_axial = sharedconv
            self.conv_coronal = sharedconv
            self.conv_sagittal = sharedconv
        elif chosen_conv == 'separateconv':
            self.conv_axial = cts.final_conv()
            self.conv_coronal = cts.final_conv()
            self.conv_sagittal = cts.final_conv()
        
        #Set up FC layers
        self.fc_axial = nn.Conv2d(16, n_outputs, kernel_size = (6,6), stride=(6,6), padding=0)
        self.fc_coronal = nn.Conv2d(16, n_outputs, kernel_size = (6,6), stride=(6,6), padding=0)
        self.fc_sagittal = nn.Conv2d(16, n_outputs, kernel_size = (6,6), stride=(6,6), padding=0)
        
        #Set up average pooling across slices
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
        
        #Set up combination function
        print('Using',chosen_comb_func,'combination function')
        assert chosen_comb_func in ['avg','max','weighted_avg']
        if 'avg' in chosen_comb_func:
            self.comb_avg = nn.AvgPool1d(kernel_size=3)
        elif chosen_comb_func == 'max':
            self.comb_max = nn.MaxPool1d(kernel_size=3)
        if chosen_comb_func == 'weighted_avg':
            #init to all ones and after applying softmax the values will all be
            #one third so it'll start out equally weighting the views
            self.comb_weights = nn.Parameter(torch.ones((n_outputs,3), dtype=torch.float32),requires_grad=True)
            self.softmax = nn.Softmax(dim=1)
        
    def extract_view_representation(self, x, conv_layers, fc_layer):
        x = cts.reshape_x(x, self.slices)
        x = self.features(x) #out shape [slices,512,14,14]
        x = conv_layers(x) #out shape [slices, 16, 6, 6]
        x = fc_layer(x) #out shape [slices,83,1,1]
        x = torch.squeeze(x) #out shape [slices, 83]
        x = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x = self.avgpool_1d(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x
    
    def apply_chosen_comb_func(self,axial,coronal,sagittal):
        all_views = torch.cat((axial,coronal,sagittal),dim=0).transpose(0,1).unsqueeze(0) #out shape [1,83,3]
        if self.chosen_comb_func == 'avg':
            return self.comb_avg(all_views) #out shape [1,83,1]
        elif self.chosen_comb_func == 'max':
            return self.comb_max(all_views) #out shape [1,83,1]
        elif self.chosen_comb_func == 'weighted_avg':
            weight_fractions = self.softmax(self.comb_weights) 
            return self.comb_avg(torch.mul(weight_fractions,all_views)) #out shape [1,83,1]
    
    def forward(self, x):
        axial = self.extract_view_representation(x['axial'],self.conv_axial,self.fc_axial)
        coronal = self.extract_view_representation(x['coronal'],self.conv_coronal,self.fc_coronal)
        sagittal = self.extract_view_representation(x['sagittal'],self.conv_sagittal,self.fc_sagittal)
        preds = self.apply_chosen_comb_func(axial,coronal,sagittal) #out shape [1,83,1]
        return preds.squeeze(dim=2) #out shape [1,83]

##############
# DEPRECATED #------------------------------------------------------------------
##############
class Body_Cll_Avg_Multiview(nn.Module):
    """(1) ResNet18
       (2) conv_collapse [512, 14, 14]->[n_outputs, 1, 1]
       (3) avg pooling over slices"""
    def __init__(self, n_outputs):
        super(Body_Cll_Avg_Multiview, self).__init__()
        self.slices = 15 #9 projections
        self.features = cts.resnet_features()
        self.conv2d_axial = cts.conv_collapse(n_outputs)
        self.conv2d_coronal = cts.conv_collapse(n_outputs)
        self.conv2d_sagittal = cts.conv_collapse(n_outputs)
        self.avgpool_1d_slices = nn.AvgPool1d(kernel_size=self.slices)
        self.avgpool_1d_views = nn.AvgPool1d(kernel_size=3)
        
    def extract_view_representation(self, x, conv2d_view):
        x = cts.reshape_x(x, self.slices)
        x = self.features(x) #out shape [slices,512,14,14]
        x = conv2d_view(x) #out shape [slices,83,1,1]
        x = torch.squeeze(x) #out shape [slices, 83]
        x = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x = self.avgpool_1d_slices(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x
    
    def forward(self, x):
        axial = self.extract_view_representation(x['axial'], self.conv2d_axial) #out shape [1, 83]
        coronal = self.extract_view_representation(x['coronal'], self.conv2d_coronal) #out shape [1, 83]
        sagittal = self.extract_view_representation(x['sagittal'], self.conv2d_sagittal) #out shape [1, 83]
        all_views = torch.cat((axial,coronal,sagittal),0).unsqueeze(0).transpose(1,2) #out shape [1, 83, 3]
        pooled = self.avgpool_1d_views(all_views) #out shape [1, 83, 1]
        return torch.squeeze(pooled, dim=2)