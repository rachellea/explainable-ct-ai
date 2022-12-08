#custom_models_combined.py
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
import math #needed for calculation of weight and bias initialization
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

class BodyCombined1(nn.Module): 
    """Combines innovations from multiple models:
        custom_models_diseasereps.BodyLocationAttn3
        custom_models_multiview.BodyAvgMultiview (sharedconv, avg)   
    """
    def __init__(self, n_outputs_lung, n_outputs_heart):
        super(BodyCombined1, self).__init__()
        self.slices = 15 #9 projections
        self.n_outputs = (2*n_outputs_lung)+n_outputs_heart
        self.n_outputs_lung = n_outputs_lung
        self.n_outputs_heart = n_outputs_heart
        self.features = cts.resnet_features()
        self.conv2d = cts.final_conv()
        
        #Calculate the spatial attention based on center slices
        #The calculation is different depending on the view
        self.heart_attn_axial = nn.Sequential(nn.Linear(3*16*6*6, 6*6), nn.Softmax())
        self.left_lung_attn_axial = nn.Sequential(nn.Linear(3*16*6*6, 6*6), nn.Softmax())
        self.right_lung_attn_axial = nn.Sequential(nn.Linear(3*16*6*6, 6*6), nn.Softmax())
        
        self.heart_attn_coronal = nn.Sequential(nn.Linear(3*16*6*6, 6*6), nn.Softmax())
        self.left_lung_attn_coronal = nn.Sequential(nn.Linear(3*16*6*6, 6*6), nn.Softmax())
        self.right_lung_attn_coronal = nn.Sequential(nn.Linear(3*16*6*6, 6*6), nn.Softmax())
        
        self.heart_attn_sagittal = nn.Sequential(nn.Linear(3*16*6*6, 6*6), nn.Softmax())
        self.left_lung_attn_sagittal = nn.Sequential(nn.Linear(3*16*6*6, 6*6), nn.Softmax())
        self.right_lung_attn_sagittal = nn.Sequential(nn.Linear(3*16*6*6, 6*6), nn.Softmax())
        
        #Final disease-specific FC layers that are also distinct based
        #on the view:
        self.fclayers_weights_axial, self.fclayers_biases_axial = init_disease_fc_layers(self.n_outputs)
        self.fclayers_weights_coronal, self.fclayers_biases_coronal = init_disease_fc_layers(self.n_outputs)
        self.fclayers_weights_sagittal, self.fclayers_biases_sagittal = init_disease_fc_layers(self.n_outputs)
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
        
        #comb_avg function for combining axial, coronal, and sagittal views
        #Each view produces 83 predictions. These 83 predictions are then
        #averaged together to get the overall model output
        self.comb_avg = nn.AvgPool1d(kernel_size=3)
    
    def extract_view_representation(self, x, fclayers_weights, fclayers_biases,
                                    heart_attn, left_lung_attn, right_lung_attn):
        x = cts.reshape_x(x, self.slices)
        x = self.features(x) #out shape [slices,512,14,14]
        x = self.conv2d(x) #out shape [slices, 16, 6, 6]
        
        #Calculate the attention maps based on the center slices
        #Use slices 6, 7, and 8 because these are in the exact center and
        #also have the highest attention weight when you do height attention.
        center_slices = x[6:9,:,:,:] #out shape [3, 16, 6, 6]
        center_slices_flat = center_slices.flatten().unsqueeze(dim=0) #out shape [1,1728]
        heart_spatial = heart_attn(center_slices_flat).reshape(1,1,1,6,6) #out shape [1,1,1,6,6]
        left_lung_spatial = left_lung_attn(center_slices_flat).reshape(1,1,1,6,6) #out shape [1,1,1,6,6]
        right_lung_spatial = right_lung_attn(center_slices_flat).reshape(1,1,1,6,6) #out shape [1,1,1,6,6]
        
        #Repeat x
        x = x.repeat(self.n_outputs,1,1,1,1) #out shape [83, slices, 16, 6, 6]
        x = x.transpose(0,1) #out shape [slices, 83, 16, 6, 6]
        
        #Apply the attention maps
        #Must follow ground truth label order, which is heart, left_lung, right_lung
        x_heart = torch.mul(x[:,0:self.n_outputs_heart,:,:,:],heart_spatial)
        x_left_lung = torch.mul(x[:,self.n_outputs_heart:self.n_outputs_heart+self.n_outputs_lung,:,:,:],left_lung_spatial)
        x_right_lung = torch.mul(x[:,-1*self.n_outputs_lung:,:,:,:],right_lung_spatial)
        x = torch.cat((x_heart,x_left_lung,x_right_lung),dim=1) #out shape [slices, 83, 16, 6, 6]
        
        x = x.flatten(start_dim=2,end_dim=4) #out shape [slices, 83, 16*6*6] = [slices, 83, 576]
        slice_preds = apply_disease_fc_layers(x, fclayers_weights, fclayers_biases)
        x = slice_preds.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x = self.avgpool_1d(x) #out shape [1, 83, 1]
        #TODO NOTE: if I were to do avgmax, then I would need to also take the
        #max here, and return that as a separate thing!
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x

    def forward(self, x):
        axial = self.extract_view_representation(x['axial'],
            self.fclayers_weights_axial,self.fclayers_biases_axial,
            heart_attn = self.heart_attn_axial,
            left_lung_attn = self.left_lung_attn_axial,
            right_lung_attn = self.right_lung_attn_axial)
        coronal = self.extract_view_representation(x['coronal'],
            self.fclayers_weights_coronal,self.fclayers_biases_coronal,
            heart_attn = self.heart_attn_coronal,
            left_lung_attn = self.left_lung_attn_coronal,
            right_lung_attn = self.right_lung_attn_coronal)
        sagittal = self.extract_view_representation(x['sagittal'],
            self.fclayers_weights_sagittal,self.fclayers_biases_sagittal,
            heart_attn = self.heart_attn_sagittal,
            left_lung_attn = self.left_lung_attn_sagittal,
            right_lung_attn = self.right_lung_attn_sagittal)
        all_views = torch.cat((axial,coronal,sagittal),dim=0).transpose(0,1).unsqueeze(0) #out shape [1,83,3]
        preds = self.comb_avg(all_views) #out shape [1,83,1]
        return preds.squeeze(dim=2) #out shape [1,83]

#############
# Functions #-------------------------------------------------------------------
#############
def init_disease_fc_layers(n_outputs):
    """Return the weights and biases of <n_outputs> disease-specific
    fully connected layers"""
    #dzfclayers_weights holds the weights for each disease-specific fc layer.
    #https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py#L40
    fclayers_weights_list = []
    fclayers_biases_list = []
    for layernum in range(n_outputs):
        #kaiming uniform init following https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
        #In order to be equivalent to the initialization of the final
        #conv2d layer in the baseline model, the fan_in used should be 576.
        #That is what we'll get in the calculation because in_features
        #is 16*6*6=576, and the weights are defined as weight = Parameter(torch.Tensor(out_features, in_features))
        #>>> nn.init._calculate_fan_in_and_fan_out(torch.rand(1,16*6*6))
        #(576, 1)
        in_features = 16*6*6
        out_features = 1
        #weight:
        weight = torch.Tensor(out_features, in_features)
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        #bias:
        bias = torch.Tensor(out_features)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        assert fan_in == 576 #sanity check based on my calculations
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(bias, -bound, bound)
        fclayers_weights_list.append(weight)
        fclayers_biases_list.append(bias)
    fclayers_weights = nn.Parameter(torch.cat(fclayers_weights_list,dim=0)) #shape [83, 576]
    fclayers_biases = nn.Parameter(torch.cat(fclayers_biases_list,dim=0)) #shape [83]
    return fclayers_weights, fclayers_biases

def apply_disease_fc_layers(x, fclayers_weights, fclayers_biases):
    """Apply the disease-specific fully connected layers"""
    slice_preds_list = []
    for slice_num in range(x.shape[0]):
        slice_data = x[slice_num,:,:] #out shape [83, 576]
        #apply all the disease-specific FC layers at once
        #Weight multiplication
        #element-wise multiply and then sum over the columns (because this
        #is equivalent to doing vector-vector multiplication between
        #the rows of slice_data and the corresponding rows of self.fclayers_weights)
        temp1 = torch.mul(slice_data,fclayers_weights) #out shape [83, 576]
        temp2 = torch.sum(temp1,dim=1) #out shape [83]
        #Bias addition
        temp3 = (temp2+fclayers_biases).unsqueeze(0) #out shape [1,83]
        #Now we have our 83 disease predictions for this slice.
        #Append these slice predictions to our list:
        slice_preds_list.append(temp3)
    slice_preds = torch.cat(slice_preds_list,dim=0) #out shape [slices, 83]
    return slice_preds
    