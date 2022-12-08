#model_outputs_classes.py
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

#########################
# Model Outputs Classes #-------------------------------------------------------
#########################
def return_modeloutputsclass(model_name):
    if model_name in ['AxialNet','AxialNet_Mask']:
        return ModelOutputs_AxialNet
    elif model_name == 'CTNetModel':
        return ModelOutputs_CTNetModel
    elif model_name == 'BodyConv':
        return ModelOutputs_BodyConv
    elif model_name == 'NoCustomConv': #ablated version of AxialNet
        return ModelOutputs_NoCustomConv
    elif model_name == 'ThreeDConv':
        return ModelOutputs_ThreeDConv
    elif model_name == 'BodyCAM':
        return ModelOutputs_BodyCAM

################################################################################
################################################################################
################################################################################
class ModelOutputs_AxialNet():
    """Class for running a AxialNet  <model> and:
       (1) Extracting activations from intermediate target layers
       (2) Extracting gradients from intermediate target layers
       (3) Returning the final model output
    
    Assumes that the <target_layer_name> is from the convolutional part of
    the model (not the ResNet feature extractor)"""
    def __init__(self, model, target_layer_name):
        self.model = model
        assert isinstance(target_layer_name,str)
        self.target_layer_name = target_layer_name
        assert self.target_layer_name == '7', 'Target layer name for last conv layer in AxialNet Model should be 7. Remove this assertion if you want to visualize a different layer.'
        #Dict where the key is the name and the value is the gradient (hook)
        self.gradients = []
        self.gradient_names = []
        self.verbose = False
    
    def save_gradient(self, grad):
        self.gradients.append(grad)
    
    def get_gradients(self):
        gradients_dict = {}
        for idx in range(len(self.gradient_names)):
            name = self.gradient_names[idx]
            grad = self.gradients[idx]
            gradients_dict[name] = grad
        return gradients_dict
    
    def run_model(self, x):
        """Run the model self.model on the input <x> and return the activations
        and output"""
        #x initially is shape 1, 135, 3, 420, 420]
        x = x.squeeze(dim=0) #[135, 3, 420, 420]
        assert list(x.shape)[1:]==[3,420,420]
        
        #Dict where the key is the name and the value is the activation
        target_activations = {}
        
        #Set up layers of the model to call
        features = self.model.features._modules.items()
        conv2d = self.model.conv2d._modules.items()
        fc = self.model.fc
        avgpool_1d = self.model.avgpool_1d
        
        #Iterate through first part of model:
        for name, module in features:
            if self.verbose: print('Applying features layer',name,'to data')
            x = module(x)
        
        #Iterate through second part of model, conv2d:
        #This is where the target activations and gradients come from. 
        for name, module in conv2d:
            if self.verbose: print('Applying conv2d layer',name,'to data')
            x = module(x)
            if self.verbose: print('\tdata shape after applying layer:',x.shape)
            if name == self.target_layer_name: #names are e.g. '4'. target_layer_name can be e.g. ['2','4']
                x.register_hook(self.save_gradient)
                self.gradient_names.append(name)
                target_activations[name] = x.cpu().data
        
        #Apply the rest of the model to get the final output
        if self.verbose: print('Applying FC and avg pool')
        x = fc(x) #[slices, 83, 1, 1]
        x = torch.squeeze(x) #out shape [slices, 83]
        x_perslice_scores = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]. Scores for 83 diseases on every slice
        x = avgpool_1d(x_perslice_scores) #out shape [1, 83, 1]
        output = torch.squeeze(x, dim=2) #out shape [1, 83]
        return target_activations, x_perslice_scores, output

#AxialNet conv2d layer numbers:
# (conv2d): Sequential(
#     (0): Conv2d(512, 64, kernel_size=(3, 3), stride=(1, 1))
#     (1): ReLU(inplace)
#     (2): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1))
#     (3): ReLU(inplace)
#     (4): Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1))
#     (5): ReLU(inplace)
#     (6): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1))
#     (7): ReLU(inplace)
#   )

################################################################################
################################################################################
################################################################################
class ModelOutputs_NoCustomConv():
    """Class for running a AxialNet  <model> where the custom convolutional
    layers have been ablated:
       (1) Extracting activations from intermediate target layers
       (2) Extracting gradients from intermediate target layers
       (3) Returning the final model output
    
    Assumes that the <target_layer_name> is from the end of the ResNet18 because
    that is the last convolutional part of this ablated model"""
    def __init__(self, model, target_layer_name):
        self.model = model
        assert isinstance(target_layer_name,str)
        self.target_layer_name = target_layer_name
        assert self.target_layer_name == '7', 'Target layer name for last conv layer in AxialNet Model should be 7. Remove this assertion if you want to visualize a different layer.'
        #Dict where the key is the name and the value is the gradient (hook)
        self.gradients = []
        self.gradient_names = []
        self.verbose = False
    
    def save_gradient(self, grad):
        self.gradients.append(grad)
    
    def get_gradients(self):
        gradients_dict = {}
        for idx in range(len(self.gradient_names)):
            name = self.gradient_names[idx]
            grad = self.gradients[idx]
            gradients_dict[name] = grad
        return gradients_dict
    
    def run_model(self, x):
        """Run the model self.model on the input <x> and return the activations
        and output"""
        #x initially is shape 1, 135, 3, 420, 420]
        x = x.squeeze(dim=0) #[135, 3, 420, 420]
        assert list(x.shape)[1:]==[3,420,420]
        
        #Dict where the key is the name and the value is the activation
        target_activations = {}
        
        #Set up layers of the model to call
        features = self.model.features._modules.items()
        fc = self.model.fc
        avgpool_1d = self.model.avgpool_1d
        
        #Iterate through first part of model:
        for name, module in features:
            if self.verbose: print('Applying features layer',name,'to data')
            x = module(x)
            if self.verbose: print('\tdata shape after applying layer:',x.shape)
            if name == self.target_layer_name: #names are e.g. '4'. target_layer_name can be e.g. ['2','4']
                x.register_hook(self.save_gradient)
                self.gradient_names.append(name)
                target_activations[name] = x.cpu().data
        
        #Apply the rest of the model to get the final output
        if self.verbose: print('Applying FC and avg pool')
        x = fc(x) #[slices, 83, 1, 1]
        x = torch.squeeze(x) #out shape [slices, 83]
        x_perslice_scores = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]. Scores for 83 diseases on every slice
        x = avgpool_1d(x_perslice_scores) #out shape [1, 83, 1]
        output = torch.squeeze(x, dim=2) #out shape [1, 83]
        return target_activations, x_perslice_scores, output

################################################################################
################################################################################
################################################################################
class ModelOutputs_CTNetModel():
    """Class for running a CTNetModel <model> and:
       (1) Extracting activations from intermediate target layers
       (2) Extracting gradients from intermediate target layers
       (3) Returning the final model output
    
    Assumes that the <target_layer_name> is from the end of the ResNet feature
    extractor ('features'), at the last possible layer where the 3D connection
    to the input volume is available.
    As soon as we get to the 3D convolutions over features, the spatial
    relationship between the representation and the original volume is lost."""
    def __init__(self, model, target_layer_name):
        self.model = model
        assert isinstance(target_layer_name,str)
        self.target_layer_name = target_layer_name
        assert self.target_layer_name == '7', 'Target layer name for last conv layer in CTNet Model should be 7. Remove this assertion if you want to visualize a different layer.'
        #Dict where the key is the name and the value is the gradient (hook)
        self.gradients = []
        self.gradient_names = []
        self.verbose = False
    
    def save_gradient(self, grad):
        self.gradients.append(grad)
    
    def get_gradients(self):
        gradients_dict = {}
        for idx in range(len(self.gradient_names)):
            name = self.gradient_names[idx]
            grad = self.gradients[idx]
            gradients_dict[name] = grad
        return gradients_dict
    
    def run_model(self, x):
        """Run the model self.model on the input <x> and return the activations
        and output"""
        #x initially is shape [1, 135, 3, 420, 420]
        x = x.squeeze(dim=0) #[135, 3, 420, 420]
        assert list(x.shape)[1:]==[3,420,420]
        
        #Dict where the key is the name and the value is the activation
        target_activations = {}
        
        #Set up layers of the model to call
        features = self.model.features._modules.items()
        reducingconvs = self.model.reducingconvs._modules.items()
        classifier = self.model.classifier._modules.items()
        
        #Iterate through first part of model:
        for name, module in features:
            if self.verbose: print('Applying features layer',name,'to data')
            x = module(x)
            if self.verbose: print('\tdata shape after applying layer:',x.shape)
            if name == self.target_layer_name: #names are e.g. '4'. target_layer_name can be e.g. ['2','4']
                x.register_hook(self.save_gradient)
                self.gradient_names.append(name)
                target_activations[name] = x.cpu().data
        
        #Bring back the batch size dimension
        x = x.unsqueeze(dim=0)
        
        #Iterate through second part of model, conv2d:
        #This is where the target activations and gradients come from. 
        for name, module in reducingconvs:
            if self.verbose: print('Applying reducingconvs layer',name,'to data')
            x = module(x)
        
        #Flatten
        x = x.view(1, 16*18*5*5)
        
        #Apply the rest of the model to get the final output
        for name, module in classifier:
            if self.verbose: print('Applying classifier layer',name,'to data')
            x = module(x)
        x_perslice_scores = None #just for compatibility with the output format
        output = x
        return target_activations, x_perslice_scores, output

#This is the ResNet-18 feature extractor:
# Sequential(
#   (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#   (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU(inplace=True)
#   (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
#   (4): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (5): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (downsample): Sequential(
#         (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (6): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (downsample): Sequential(
#         (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (7): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (downsample): Sequential(
#         (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
# )

################################################################################
################################################################################
################################################################################
class ModelOutputs_BodyConv():
    """Class for running a BodyConv  <model> and:
       (1) Extracting activations from intermediate target layers
       (2) Extracting gradients from intermediate target layers
       (3) Returning the final model output
    
    Assumes that the <target_layer_name> is from the convolutional part of
    the model (not the ResNet feature extractor)"""
    def __init__(self, model, target_layer_name):
        self.model = model
        assert isinstance(target_layer_name,str)
        self.target_layer_name = target_layer_name
        assert self.target_layer_name == '5', 'Target layer name for last conv layer in BodyConv Model should be 5. Remove this assertion if you want to visualize a different layer.'
        #Dict where the key is the name and the value is the gradient (hook)
        self.gradients = []
        self.gradient_names = []
        self.verbose = False
    
    def save_gradient(self, grad):
        self.gradients.append(grad)
    
    def get_gradients(self):
        gradients_dict = {}
        for idx in range(len(self.gradient_names)):
            name = self.gradient_names[idx]
            grad = self.gradients[idx]
            #See below comment on the activation map reshaping - same principle,
            #must make sure gradient is returned with dimensions in the order
            #that gradcam.py expects.
            #Right now it's [1, 16, 5, 5, 5] for [batch, features, height, width, depth]
            #needs to be [height, features, width, depth]
            grad_temp = grad.squeeze(0) #out shape [16, 5, 5, 5]
            grad_temp = grad_temp.transpose(0,1) #out shape [5, 16, 5, 5]
            gradients_dict[name] = grad_temp
        return gradients_dict
    
    def run_model(self, x):
        """Run the model self.model on the input <x> and return the activations
        and output"""
        #x initially is shape 1, 135, 3, 420, 420]
        x = x.squeeze(dim=0) #[135, 3, 420, 420]
        assert list(x.shape)[1:]==[3,420,420]
        
        #Dict where the key is the name and the value is the activation
        target_activations = {}
        
        #Set up layers of the model to call
        features = self.model.features._modules.items()
        reducingconvs = self.model.reducingconvs._modules.items()
        classifier = self.model.classifier._modules.items()
        
        #Iterate through first part of model:
        for name, module in features:
            if self.verbose: print('Applying features layer',name,'to data')
            x = module(x)
        
        #Necessary rearranging
        x = x.transpose(0,1).unsqueeze(0)
        x = x.contiguous()
        
        #Iterate through second part of model, reducingconvs:
        #This is where the target activations and gradients come from. 
        for name, module in reducingconvs:
            if self.verbose: print('Applying conv2d layer',name,'to data')
            x = module(x)
            if self.verbose: print('\tdata shape after applying layer:',x.shape)
            if name == self.target_layer_name: #name is e.g. '5'. 
                x.register_hook(self.save_gradient)
                self.gradient_names.append(name)
                #The gradcam code assumes that target_activations and gradients
                #are arranged as [height, features, width, depth]
                #for example [135, 16, 6, 6]
                #However the representation in BodyConv reducingconvs is
                #[batch, features, height, width, depth] e.g. [1,512,134,14,14]
                #is the input to the reducingconvs.
                #the shape of x at target layer '5': [1, 16, 5, 5, 5]
                x_temp = x.squeeze(0) #out shape [16, 5, 5, 5] for [features, height, width, depth]
                x_temp = x_temp.transpose(0,1) #out shape [5, 16, 5, 5] for [height, features, width, depth]
                target_activations[name] = x_temp.cpu().data
        
        #Flatten
        #the final output shape of reducingconvs is [1, 16, 5, 5, 5]
        x = x.view(1, 16*5*5*5)
        
        for name, module in classifier:
            x = module(x)
        
        output = x
        x_perslice_scores = None #for compatibility
        return target_activations, x_perslice_scores, output

# BodyConv model after the ResNet18 feature extractor:
# (reducingconvs): Sequential(
#   (0): Conv3d(512, 64, kernel_size=(3, 3, 3), stride=(3, 1, 1))
#   (1): ReLU()
#   (2): Conv3d(64, 32, kernel_size=(3, 3, 3), stride=(3, 1, 1))
#   (3): ReLU()
#   (4): Conv3d(32, 16, kernel_size=(3, 2, 2), stride=(3, 2, 2))
#   (5): ReLU()
# )
# (classifier): Sequential(
#   (0): Linear(in_features=2000, out_features=128, bias=True)
#   (1): ReLU(inplace=True)
#   (2): Dropout(p=0.5, inplace=False)
#   (3): Linear(in_features=128, out_features=96, bias=True)
#   (4): ReLU(inplace=True)
#   (5): Dropout(p=0.5, inplace=False)
#   (6): Linear(in_features=96, out_features=80, bias=True)
# )

################################################################################
################################################################################
################################################################################
class ModelOutputs_ThreeDConv():
    """Class for running a ThreeDConv  <model> and:
       (1) Extracting activations from intermediate target layers
       (2) Extracting gradients from intermediate target layers
       (3) Returning the final model output
    
    Assumes that the <target_layer_name> is from the convolutional part of
    the model"""
    def __init__(self, model, target_layer_name):
        self.model = model
        assert isinstance(target_layer_name,str)
        self.target_layer_name = target_layer_name
        assert self.target_layer_name == '11', 'Target layer name for last conv layer in Three3DConv Model should be 11. Remove this assertion if you want to visualize a different layer.'
        #Dict where the key is the name and the value is the gradient (hook)
        self.gradients = []
        self.gradient_names = []
        self.verbose = False
    
    def save_gradient(self, grad):
        self.gradients.append(grad)
    
    def get_gradients(self):
        gradients_dict = {}
        for idx in range(len(self.gradient_names)):
            name = self.gradient_names[idx]
            grad = self.gradients[idx]
            #See below comment on the activation map reshaping - same principle,
            #must make sure gradient is returned with dimensions in the order
            #that gradcam.py expects.
            #Right now it's [1, 64, 5, 5, 5] for [batch, features, height, width, depth]
            #needs to be [height, features, width, depth]
            grad_temp = grad.squeeze(0) #out shape [64, 5, 5, 5]
            grad_temp = grad_temp.transpose(0,1) #out shape [5, 64, 5, 5]
            gradients_dict[name] = grad_temp
        return gradients_dict
    
    def run_model(self, x):
        """Run the model self.model on the input <x> and return the activations
        and output"""
        #input x has shape [1,405,420,420]
        assert list(x.shape)[1:]==[405,420,420]
        
        #Dict where the key is the name and the value is the activation
        target_activations = {}
        
        #Set up layers of the model to call
        reducingconvs = self.model.reducingconvs._modules.items()
        classifier = self.model.classifier._modules.items()
        
        x = x.unsqueeze(dim=0) #get shape [1,1,405,420,420] = [N,C,D,H,W]
        
        #Iterate through second part of model, reducingconvs:
        #This is where the target activations and gradients come from. 
        for name, module in reducingconvs:
            if self.verbose: print('Applying conv2d layer',name,'to data')
            x = module(x)
            if self.verbose: print('\tdata shape after applying layer:',x.shape)
            if name == self.target_layer_name: #name is e.g. '5'. 
                x.register_hook(self.save_gradient)
                self.gradient_names.append(name)
                #The gradcam code assumes that target_activations and gradients
                #are arranged as [height, features, width, depth]
                #for example [135, 16, 6, 6]
                #However the representation in ThreeDConv reducingconvs is
                #[batch, features, height, width, depth] e.g. [1, 64, 5, 5, 5]
                #is the input to the reducingconvs.
                #the shape of x at target layer '11': [1, 64, 5, 5, 5]
                x_temp = x.squeeze(0) #out shape [64, 5, 5, 5] for [features, height, width, depth]
                x_temp = x_temp.transpose(0,1) #out shape [5, 64, 5, 5] for [height, features, width, depth]
                target_activations[name] = x_temp.cpu().data
        
        #Flatten
        #the final output shape of reducingconvs is [1, 16, 5, 5, 5]
        x = x.view(1, 64*5*5*5)
        
        for name, module in classifier:
            x = module(x)
        
        output = x
        x_perslice_scores = None #for compatibility
        return target_activations, x_perslice_scores, output

# ThreeDConv(
#   (reducingconvs): Sequential(
#     (0): Conv3d(1, 64, kernel_size=(3, 3, 3), stride=(3, 3, 3))
#     (1): ReLU()
#     (2): Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(3, 3, 3))
#     (3): ReLU()
#     (4): Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(3, 3, 3))
#     (5): ReLU()
#     (6): Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(3, 3, 3))
#     (7): ReLU()
#     (8): Conv3d(512, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1))
#     (9): ReLU()
#     (10): Conv3d(128, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1))
#     (11): ReLU()
#   )
#   (classifier): Sequential(
#     (0): Linear(in_features=8000, out_features=128, bias=True)
#     (1): ReLU()
#     (2): Dropout(p=0.1, inplace=False)
#     (3): Linear(in_features=128, out_features=96, bias=True)
#     (4): ReLU()
#     (5): Dropout(p=0.1, inplace=False)
#     (6): Linear(in_features=96, out_features=80, bias=True)
#   )
# )

################################################################################
################################################################################
################################################################################
class ModelOutputs_BodyCAM():
    """Class for running a BodyCAM <model> and:
       (1) Extracting activations from intermediate target layers
       (2) Extracting gradients from intermediate target layers
       (3) Returning the final model output
    
    Assumes that the <target_layer_name> is from the convolutional part of
    the model (not the ResNet feature extractor)"""
    def __init__(self, model, target_layer_name):
        self.model = model
        assert isinstance(target_layer_name,str)
        self.target_layer_name = target_layer_name
        assert self.target_layer_name == '7', 'Target layer name for last conv layer in BodyCAM Model should be 7. Remove this assertion if you want to visualize a different layer.'
        #Dict where the key is the name and the value is the gradient (hook)
        self.gradients = []
        self.gradient_names = []
        self.verbose = False
    
    def save_gradient(self, grad):
        self.gradients.append(grad)
    
    def get_gradients(self):
        gradients_dict = {}
        for idx in range(len(self.gradient_names)):
            name = self.gradient_names[idx]
            grad = self.gradients[idx]
            gradients_dict[name] = grad
        return gradients_dict
    
    def run_model(self, x):
        """Run the model self.model on the input <x> and return the activations
        and output"""
        #x initially is shape 1, 135, 3, 420, 420]
        x = x.squeeze(dim=0) #[135, 3, 420, 420]
        assert list(x.shape)[1:]==[3,420,420]
        
        #Dict where the key is the name and the value is the activation
        target_activations = {}
        
        #Set up layers of the model to call
        features = self.model.features._modules.items()
        conv2d = self.model.conv2d._modules.items()
        avgpool_3d = self.model.avgpool_3d
        fc = self.model.fc
        
        #Iterate through first part of model:
        for name, module in features:
            if self.verbose: print('Applying features layer',name,'to data')
            x = module(x)
        
        #Iterate through second part of model, conv2d:
        #This is where the target activations and gradients come from. 
        for name, module in conv2d:
            if self.verbose: print('Applying conv2d layer',name,'to data')
            x = module(x)
            if self.verbose: print('\tdata shape after applying layer:',x.shape)
            if name == self.target_layer_name: #name is e.g. '5'. 
                x.register_hook(self.save_gradient)
                #The gradcam code assumes that target_activations and gradients
                #are arranged as [height, features, width, depth]
                #for example [135, 16, 6, 6] which is the shape here.
                self.gradient_names.append(name)
                target_activations[name] = x.cpu().data
        
        #rearrange to [N, C, D, H, W]; kernel size is for (D, H, W)
        x = x.transpose(0,1).unsqueeze(0) #out shape [1, 16, slices, 6, 6]
        #global average pooling over all dimensions except the feature dim
        #(the activation maps are 3D, with size [D x H x W])
        x = avgpool_3d(x) #out shape [1, 16, 1, 1, 1]
        #flatten
        x = x.squeeze(2).squeeze(2).squeeze(2) #out shape [1, 16]
        #fc
        x = fc(x) #out shape [1,n_outputs]
        
        output = x
        x_perslice_scores = None #for compatibility
        return target_activations, x_perslice_scores, output

# BodyCAM model after the ResNet18 feature extractor:
#   (conv2d): Sequential(
#     (0): Conv2d(512, 64, kernel_size=(3, 3), stride=(1, 1))
#     (1): ReLU(inplace)
#     (2): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1))
#     (3): ReLU(inplace)
#     (4): Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1))
#     (5): ReLU(inplace)
#     (6): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1))
#     (7): ReLU(inplace)
#   )
#   (avgpool_3d): AvgPool3d(kernel_size=(135, 6, 6), stride=(135, 6, 6), padding=0)
#   (fc): Linear(in_features=16, out_features=80, bias=True)
# )
