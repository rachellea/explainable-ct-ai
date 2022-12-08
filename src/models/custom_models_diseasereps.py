#custom_models_diseasereps.py
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

#TODOS FOR ALL MODELS:
#TODO try with Sigmoid on the attention instead of Softmax
#TODO try with self-attention instead of fixed learned attention weights

class BodyAvgDiseaseFeatureAttn(nn.Module): #7/1/2020
    """(1) ResNet18 [slices, 512, 14, 14]
       (2) conv_final to [slices, 16, 6, 6]
       (3) Make a 'copy representation': create n_outputs number of copies
           by tiling: [slices, n_outputs, 16, 6, 6]
       (4) Element wise multiply the 'copy representation' by a learned
           weight vector of shape [1, n_outputs, 16, 1, 1]. This learned
           weight vector re-weights the features for each disease separately.
           Out shape: [slices, n_outputs, 16, 6, 6] (unchanged because we used
           element-wise multiplication with broadcasting).
        (5) Apply disease-specific FC layers which for each of the n_outputs
            diseases will transform the 16*6*6 representation into a single
            disease score. This step is analogous to the final FC layer in
            the baseline model, except that in the baseline model we can
            implement it easily with Conv2d whereas here because we have
            separate disease representations we have to do something
            trickier to implement disease-specific FC layers.
            Out shape: [slices, n_outputs]
        (6) Avg pooling over slices to get [n_outputs]"""
    def __init__(self, n_outputs):
        super(BodyAvgDiseaseFeatureAttn, self).__init__()
        self.slices = 15 #9 projections
        self.n_outputs = n_outputs
        self.features = cts.resnet_features()
        self.conv2d = cts.final_conv()
        self.dzfeatweights = nn.Parameter(torch.ones((1,n_outputs,16,1,1), dtype=torch.float32),requires_grad=True)
        self.softmax = nn.Softmax(dim=2) #make the 16 feature weights per disease add to 1
        self.fclayers_weights, self.fclayers_biases = init_stacked_fc_layers(total_independent_fc_layers = n_outputs, in_features = 16*6*6)
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
        
    def forward(self, x):
        x = cts.reshape_x(x, self.slices)
        x = self.features(x) #out shape [slices,512,14,14]
        x = self.conv2d(x) #out shape [slices, 16, 6, 6]
        #Copy the representation n_outputs number of times, so that we can
        #calculate disease-specific intermediate representations, in which
        #the features have been reweighted for each disease separately:
        x = x.repeat(self.n_outputs,1,1,1,1) #out shape [83, slices, 16, 6, 6]
        x = x.transpose(0,1) #out shape [slices, 83, 16, 6, 6]
        #Element wise multiply the copy representation by the learned weights
        #The learned weights perform the feature reweighting per disease.
        #The softmax makes the features for one disease "compete against each other"
        x = torch.mul(x,self.softmax(self.dzfeatweights)) #out shape [slices, 83, 16, 6, 6]
        #Flatten
        x = x.flatten(start_dim=2,end_dim=4) #out shape [slices, 83, 16*6*6] = [slices, 83, 576]
        #Apply disease-specific FC layers
        slice_preds = apply_disease_fc_layers(x, self.fclayers_weights, self.fclayers_biases)
        #Final steps are the same as for baseline model:
        x = slice_preds.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x = self.avgpool_1d(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x

class BodyAvg_Testing(nn.Module): #7/2/2020
    """BodyAvg model, implemented using the 'copy representation' and
    disease-specific FC layers of BodyAvgDiseaseFeatureAttn. The only purpose
    of this model is code testing: to figure out if the performance is exactly
    the same as for the BodyAvg model."""
    def __init__(self, n_outputs):
        super(BodyAvg_Testing, self).__init__()
        self.slices = 15 #9 projections
        self.n_outputs = n_outputs
        self.features = cts.resnet_features()
        self.conv2d = cts.final_conv()
        self.fclayers_weights, self.fclayers_biases = init_stacked_fc_layers(total_independent_fc_layers = n_outputs, in_features = 16*6*6)
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
        
    def forward(self, x):
        x = cts.reshape_x(x, self.slices)
        x = self.features(x) #out shape [slices,512,14,14]
        x = self.conv2d(x) #out shape [slices, 16, 6, 6]
        x = x.repeat(self.n_outputs,1,1,1,1) #out shape [83, slices, 16, 6, 6]
        x = x.transpose(0,1) #out shape [slices, 83, 16, 6, 6]
        x = x.flatten(start_dim=2,end_dim=4) #out shape [slices, 83, 16*6*6] = [slices, 83, 576]
        slice_preds = apply_disease_fc_layers(x, self.fclayers_weights, self.fclayers_biases)
        x = slice_preds.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x = self.avgpool_1d(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x

class BodyAvgDiseaseFeatureAttn2(nn.Module): #7/2/2020, updated 7/7/2020
    """See BodyAvgDiseaseFeatureAttn for more documentation including code comments.
    Difference from BodyAvgDiseaseFeatureAttn: in step (4) this model shares
    the learned feature weights between the right lung and theleft lung."""
    def __init__(self, n_outputs_lung, n_outputs_heart, nonlinearity):
        super(BodyAvgDiseaseFeatureAttn2, self).__init__()
        self.slices = 15 #9 projections
        self.n_outputs = (2*n_outputs_lung)+n_outputs_heart
        self.n_outputs_lung = n_outputs_lung
        self.n_outputs_heart = n_outputs_heart
        self.features = cts.resnet_features()
        self.conv2d = cts.final_conv()
        self.dzfeatweights_lung = nn.Parameter(torch.ones((1,n_outputs_lung,16,1,1), dtype=torch.float32),requires_grad=True)
        self.dzfeatweights_heart = nn.Parameter(torch.ones((1,n_outputs_heart,16,1,1), dtype=torch.float32),requires_grad=True)
        #Nonlinearity that gets applied to the feature weighting:
        if nonlinearity == 'softmax':
            self.nonlinearity = nn.Softmax(dim=2) #make the 16 feature weights per disease add to 1
        elif nonlinearity == 'sigmoid':
            self.nonlinearity = nn.Sigmoid()
        self.fclayers_weights, self.fclayers_biases = init_stacked_fc_layers(total_independent_fc_layers = self.n_outputs, in_features = 16*6*6)
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
        
    def forward(self, x):
        x = cts.reshape_x(x, self.slices)
        x = self.features(x) #out shape [slices,512,14,14]
        x = self.conv2d(x) #out shape [slices, 16, 6, 6]
        x = x.repeat(self.n_outputs,1,1,1,1) #out shape [83, slices, 16, 6, 6]
        x = x.transpose(0,1) #out shape [slices, 83, 16, 6, 6]
        
        #Apply the feature weights.
        #Must follow ground truth label order, which is heart, left_lung, right_lung
        x_heart = torch.mul(x[:,0:self.n_outputs_heart,:,:,:],self.nonlinearity(self.dzfeatweights_heart))
        x_left_lung = torch.mul(x[:,self.n_outputs_heart:self.n_outputs_heart+self.n_outputs_lung,:,:,:],self.nonlinearity(self.dzfeatweights_lung))
        x_right_lung = torch.mul(x[:,-1*self.n_outputs_lung:,:,:,:],self.nonlinearity(self.dzfeatweights_lung))
        x = torch.cat((x_heart,x_left_lung,x_right_lung),dim=1) #out shape [slices, 83, 16, 6, 6]
        
        x = x.flatten(start_dim=2,end_dim=4) #out shape [slices, 83, 16*6*6] = [slices, 83, 576]
        slice_preds = apply_disease_fc_layers(x, self.fclayers_weights, self.fclayers_biases)
        x = slice_preds.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x = self.avgpool_1d(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x

class BodyLocationAttn3(nn.Module): #7/2/2020, updated 7/7/2020
    """See BodyAvgDiseaseFeatureAttn for more documentation including code comments.
    Difference from BodyAvgDiseaseFeatureAttn: uses spatial attention instead of
    feature attention. Specifically there is right lung, heart, and left lung
    spatial attention. Also, instead of being fixed weights every time, the
    weights are learned based on using the center slices (since the center
    slices are most indicative of where the right lung, heart, and left
    lung are located.) So this is trainable soft self-attention."""
    def __init__(self, n_outputs_lung, n_outputs_heart, nonlinearity):
        super(BodyLocationAttn3, self).__init__()
        self.slices = 15 #9 projections
        self.n_outputs = (2*n_outputs_lung)+n_outputs_heart
        self.n_outputs_lung = n_outputs_lung
        self.n_outputs_heart = n_outputs_heart
        self.features = cts.resnet_features()
        self.conv2d = cts.final_conv()
        #Calculate the spatial attention based on center slices
        if nonlinearity == 'softmax':
            chosen_nonlinearity = nn.Softmax()
        elif nonlinearity == 'sigmoid':
            chosen_nonlinearity = nn.Sigmoid()
        
        self.heart_attn_fc = nn.Sequential(nn.Linear(3*16*6*6, 6*6),
                                           chosen_nonlinearity)
        self.left_lung_attn_fc = nn.Sequential(nn.Linear(3*16*6*6, 6*6),
                                           chosen_nonlinearity)
        self.right_lung_attn_fc = nn.Sequential(nn.Linear(3*16*6*6, 6*6),
                                           chosen_nonlinearity)
        self.fclayers_weights, self.fclayers_biases = init_stacked_fc_layers(total_independent_fc_layers = self.n_outputs, in_features = 16*6*6)
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
        
    def forward(self, x):
        x = cts.reshape_x(x, self.slices)
        x = self.features(x) #out shape [slices,512,14,14]
        x = self.conv2d(x) #out shape [slices, 16, 6, 6]
        
        #Calculate the attention maps based on the center slices
        #Use slices 6, 7, and 8 because these are in the exact center and
        #also have the highest attention weight when you do height attention.
        center_slices = x[6:9,:,:,:] #out shape [3, 16, 6, 6]
        center_slices_flat = center_slices.flatten().unsqueeze(dim=0) #out shape [1,1728]
        self.heart_spatial = self.heart_attn_fc(center_slices_flat).reshape(1,1,1,6,6) #out shape [1,1,1,6,6]
        self.left_lung_spatial = self.left_lung_attn_fc(center_slices_flat).reshape(1,1,1,6,6) #out shape [1,1,1,6,6]
        self.right_lung_spatial = self.right_lung_attn_fc(center_slices_flat).reshape(1,1,1,6,6) #out shape [1,1,1,6,6]
        
        #Repeat x
        x = x.repeat(self.n_outputs,1,1,1,1) #out shape [83, slices, 16, 6, 6]
        x = x.transpose(0,1) #out shape [slices, 83, 16, 6, 6]
        
        #Apply the attention maps
        #Must follow ground truth label order, which is heart, left_lung, right_lung
        x_heart = torch.mul(x[:,0:self.n_outputs_heart,:,:,:],self.heart_spatial)
        x_left_lung = torch.mul(x[:,self.n_outputs_heart:self.n_outputs_heart+self.n_outputs_lung,:,:,:],self.left_lung_spatial)
        x_right_lung = torch.mul(x[:,-1*self.n_outputs_lung:,:,:,:],self.right_lung_spatial)
        x = torch.cat((x_heart,x_left_lung,x_right_lung),dim=1) #out shape [slices, 83, 16, 6, 6]
        
        x = x.flatten(start_dim=2,end_dim=4) #out shape [slices, 83, 16*6*6] = [slices, 83, 576]
        slice_preds = apply_disease_fc_layers(x, self.fclayers_weights, self.fclayers_biases)
        x = slice_preds.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x = self.avgpool_1d(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x

class BodyDiseaseSpatialAttn4(nn.Module): #7/7/2020 #TODO test this
    """See BodyAvgDiseaseFeatureAttn for more documentation including code comments.
    Difference from BodyLocationAttn3: while 4 also uses spatial
    attention (like 3), 4 does spatial attention per disease instead of per
    location."""
    def __init__(self, n_outputs, nonlinearity):
        super(BodyDiseaseSpatialAttn4, self).__init__()
        self.slices = 15 #9 projections
        self.n_outputs = n_outputs
        self.features = cts.resnet_features()
        self.conv2d = cts.final_conv()
        #Calculate the spatial attention based on center slices
        if nonlinearity == 'softmax':
            self.nonlinearity = nn.Softmax(dim=2)
        elif nonlinearity == 'sigmoid':
            self.nonlinearity = nn.Sigmoid()
        #FC layers for calculating the disease-specific spatial attention
        #For each disease and each element of the 6x6 I learn a different FC layer:
        self.fcattns_weights, self.fcattns_biases = init_stacked_fc_layers(total_independent_fc_layers = n_outputs*6*6, in_features = 16)
        #FC layers for calculating the final disease predictions
        self.fclayers_weights, self.fclayers_biases = init_stacked_fc_layers(total_independent_fc_layers = n_outputs, in_features = 16*6*6)
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
        
    def forward(self, x):
        x = cts.reshape_x(x, self.slices)
        x = self.features(x) #out shape [slices,512,14,14]
        x = self.conv2d(x) #out shape [slices, 16, 6, 6]
        x = x.repeat(self.n_outputs,1,1,1,1) #out shape [83, slices, 16, 6, 6]
        x = x.transpose(0,1) #out shape [slices, 83, 16, 6, 6]
        
        #Calculate the disease-specific spatial attention:
        attn_raw_list = []
        for slice_num in range(self.slices):
            slice_data = x[slice_num,:,:,:,:] #out shape [83, 16, 6, 6]
            slice_data = slice_data.flatten(start_dim=2,end_dim=3).transpose(1,2) #out shape [83, 6*6, 16]
            slice_data = slice_data.flatten(start_dim=0,end_dim=1) #out shape [83*6*6, 16]
            temp1 = torch.mul(slice_data,self.fcattns_weights) #out shape [83*6*6, 16]
            temp2 = torch.sum(temp1,dim=1) #out shape [83*6*6]
            temp3 = (temp2+self.fcattns_biases).unsqueeze(0) #out shape [83*6*6]
            attn_raw_list.append(temp3)
        attn_raw = torch.cat(attn_raw_list,dim=0) #out shape [slices, 83*6*6]
        attn_raw = torch.reshape(attn_raw,(self.slices,self.n_outputs,6*6)) #out shape [slices, 83, 6*6]
        attn = self.nonlinearity(attn_raw) #out shape [slices, 83, 6*6]
        attn = torch.reshape(attn,(self.slices,self.n_outputs,6,6)).unsqueeze(2) #out shape [slices, 83, 1, 6, 6]
        
        #Apply the attention
        x_times_attn = torch.mul(x, attn) #out shape [slices, 83, 16, 6, 6]
        
        #Disease predictions
        x = x.flatten(start_dim=2,end_dim=4) #out shape [slices, 83, 16*6*6] = [slices, 83, 576]
        slice_preds = apply_disease_fc_layers(x, self.fclayers_weights, self.fclayers_biases)
        x = slice_preds.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x = self.avgpool_1d(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x

class BodyDiseaseSpatialAttn5(nn.Module): #7/7/2020 #TODO test this
    """See BodyAvgDiseaseFeatureAttn for more documentation including code comments.
    Difference from BodyDiseaseSpatialAttn4: whereas 4 learns a different
    mapping of 16 features -> 1 spatial attn value for each element of the 6x6
    square, 5 uses a convolution layer such that the mapping of 16 -> 1 is
    the same for each element of the 6x6 square"""
    def __init__(self, n_outputs, nonlinearity):
        super(BodyDiseaseSpatialAttn5, self).__init__()
        self.slices = 15 #9 projections
        self.n_outputs = n_outputs
        self.features = cts.resnet_features()
        self.conv2d = cts.final_conv()
        #Calculate the spatial attention based on center slices
        if nonlinearity == 'softmax':
            self.nonlinearity = nn.Softmax(dim=2)
        elif nonlinearity == 'sigmoid':
            self.nonlinearity = nn.Sigmoid()
        
        #Conv layer for calculating the disease-specific spatial attention
        #For each disease and each element of the 6x6 I learn a different FC layer:
        self.attn_conv = nn.Sequential(
            nn.Conv2d(16, 83, kernel_size = (1,1), stride=(1,1), padding=0),
            self.nonlinearity)
        
        #FC layers for calculating the final disease predictions
        self.fclayers_weights, self.fclayers_biases = init_stacked_fc_layers(total_independent_fc_layers = n_outputs, in_features = 16*6*6)
        self.avgpool_1d = nn.AvgPool1d(kernel_size=self.slices)
        
    def forward(self, x):
        x = cts.reshape_x(x, self.slices)
        x = self.features(x) #out shape [slices,512,14,14]
        x = self.conv2d(x) #out shape [slices, 16, 6, 6]
        
        #Calculate the disease-specific spatial attention:
        attn = self.attn_conv(x).unsqueeze(2) #out shape [slices, 83, 1, 6, 6]
        
        #Apply the attention
        x = x.repeat(self.n_outputs,1,1,1,1) #out shape [83, slices, 16, 6, 6]
        x = x.transpose(0,1) #out shape [slices, 83, 16, 6, 6]
        x_times_attn = torch.mul(x, attn) #out shape [slices, 83, 16, 6, 6]
        
        #Disease predictions
        x = x.flatten(start_dim=2,end_dim=4) #out shape [slices, 83, 16*6*6] = [slices, 83, 576]
        slice_preds = apply_disease_fc_layers(x, self.fclayers_weights, self.fclayers_biases)
        x = slice_preds.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]
        x = self.avgpool_1d(x) #out shape [1, 83, 1]
        x = torch.squeeze(x, dim=2) #out shape [1, 83]
        return x

#############
# Functions #-------------------------------------------------------------------
#############
def init_stacked_fc_layers(total_independent_fc_layers, in_features):
    """Return the weights and biases of <total_independent_fc_layers>
    fully connected layers.
    Let's say there are 83 <total_independent_fc_layers> and there are
    16*6*6 in_features. Then the produced fclayers_weights will have shape
    83 x 576 and the produced fclayers_biases will have shape 83.
    Each row corresponds to one FC layer that goes from a 1 x 576 representation
    to a 1."""
    #dzfclayers_weights holds the weights for each disease-specific fc layer.
    #https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py#L40
    fclayers_weights_list = []
    fclayers_biases_list = []
    out_features = 1
    for layernum in range(total_independent_fc_layers):
        #kaiming uniform init following https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
        #for the case where we are doing disease-specific FC layers (i.e.
        #where total_independent_fc_layers = 83 and in_features = 16*6*6)
        #in order to be equivalent to the initialization of the final
        #conv2d layer in the baseline model, the fan_in used should be 576.
        #That is what we'll get in the calculation because in_features
        #is 16*6*6=576, and the weights are defined as weight = Parameter(torch.Tensor(out_features, in_features))
        #>>> nn.init._calculate_fan_in_and_fan_out(torch.rand(1,16*6*6))
        #(576, 1)
        #weight:
        weight = torch.Tensor(out_features, in_features)
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        #bias:
        bias = torch.Tensor(out_features)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        assert fan_in == in_features #e.g. 576 for in_features = 16*6*6. sanity check based on my calculations
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(bias, -bound, bound)
        fclayers_weights_list.append(weight)
        fclayers_biases_list.append(bias)
    fclayers_weights = nn.Parameter(torch.cat(fclayers_weights_list,dim=0)) #e.g. shape [83, 576]
    fclayers_biases = nn.Parameter(torch.cat(fclayers_biases_list,dim=0)) #e.g. shape [83]
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
    