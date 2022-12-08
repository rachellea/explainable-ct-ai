#specific gradcam

import numpy as np
import torch, torch.nn as nn

import gradcam

from load_dataset import custom_datasets
from models import custom_models_base

class ModelOutputs_BodyAvg():
    """Class for running a BodyAvg  <model> and:
       (1) Extracting activations from intermediate target layers
       (2) Extracting gradients from intermediate target layers
       (3) Returning the final model output
    
    Assumes that the <target_layers> are from the convolutional part of
    the model (not the ResNet feature extractor)"""
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        #Dict where the key is the name and the value is the gradient (hook)
        self.gradients = []
        self.gradient_names = []
    
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
        assert list(x.shape)==[45,3,420,420] #45 for whole data
        
        #Dict where the key is the name and the value is the activation
        target_activations = {}
        
        #Set up layers of the model to call
        features = self.model.features._modules.items()
        conv2d = self.model.conv2d._modules.items()
        fc = self.model.fc
        avgpool_1d = self.model.avgpool_1d
        
        #Iterate through first part of model:
        for name, module in features:
            print('Applying features layer',name,'to data')
            x = module(x)
        
        #Iterate through second part of model, conv2d:
        #This is where the target activations and gradients come from. 
        for name, module in conv2d:
            print('Applying conv2d layer',name,'to data')
            x = module(x)
            print('\tdata shape after applying layer:',x.shape)
            if name in self.target_layers: #names are e.g. '4'. target_layers can be e.g. ['2','4']
                x.register_hook(self.save_gradient)
                self.gradient_names.append(name)
                target_activations[name] = x.cpu().data.numpy() #TODO is it okay to transfer to CPU here or do I have to wait? i.e. does it make a copy (in which case it is ok) or does it delete from gpu?
        
        #Apply the rest of the model to get the final output
        print('Applying FC and avg pool')
        x = fc(x) #[slices, 83, 1, 1]
        x = torch.squeeze(x) #out shape [slices, 83]
        x_perslice_scores = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]. Scores for 83 diseases on every slice
        x = avgpool_1d(x_perslice_scores) #out shape [1, 83, 1]
        output = torch.squeeze(x, dim=2) #out shape [1, 83]
        return target_activations, x_perslice_scores, output

#Body_Avg conv2d layer numbers:
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

if __name__ == '__main__':
    gradcam.RunGradCam(descriptor='BodyAvgAllWholeData',
                gradcamclass = gradcam.GradCam,
                modeloutputsclass = ModelOutputs_BodyAvg,
                custom_net = custom_models_base.BodyAvg,
                custom_net_args = {'n_outputs':83,'slices':45},
                params_path = '/home/rlb61/data/img-hiermodel2/results/2020-08/2020-08-06_BodyAvgAllWholeData-OLDCODE06/params/BodyAvgAllWholeData',
                dataset_class = custom_datasets.CTDataset_2019_10,
                target_layer_name = '6', #e.g. ['3','4']
                dataset_args = {'label_type_ld':'location_disease_0323',
                                    'label_meanings':'all', #can be 'all' or a list of strings
                                    'num_channels':3,
                                    'pixel_bounds':[-1000,200],
                                    'max_slices':135, #no projections since they aren't all saved locally
                                    'max_side_length':420,
                                    'data_augment':True,
                                    'crop_type':'single',
                                    'selected_note_acc_files':{'train':'/home/rlb61/data/img-hiermodel2/load_dataset/ground_truth/predefined_subsets/2020-01-10-imgtrain_random2000.csv',
                                                           'valid':'/home/rlb61/data/img-hiermodel2/load_dataset/ground_truth/predefined_subsets/2020-01-10-imgvalid_a_random1000.csv'},
                                    'from_seg':False,
                                    'view':'axial',
                                    'projections9':False})
        