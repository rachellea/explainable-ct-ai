#specific gradcam

import os
import numpy as np
import torch, torch.nn as nn

import gradcam

from load_dataset import custom_datasets
from models import custom_models_multiview
from models import components as cts

class ModelOutputs_BodyAvgMultiview():
    """Class for running a BodyAvgMultiview <model> and:
       (1) Extracting activations from intermediate target layers
       (2) Extracting gradients from intermediate target layers
       (3) Returning the final model output
    Assumes that the <target_layers> are from the convolutional part of
    the model (not the ResNet feature extractor)"""
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        #Dict where the key is the name and the value is the gradient (hook)
        self.gradients_axial = []
        self.gradient_names_axial = []
        self.gradients_coronal = []
        self.gradient_names_coronal = []
        self.gradients_sagittal = []
        self.gradient_names_sagittal = []
    
    def save_gradient_axial(self, grad):
        self.gradients_axial.append(grad)
    
    def save_gradient_coronal(self, grad):
        self.gradients_coronal.append(grad)
    
    def save_gradient_sagittal(self, grad):
        self.gradients_sagittal.append(grad)
    
    def get_gradients(self):
        all_view_dicts = {'axial':self.get_gradients_helper(self.gradients_axial, self.gradient_names_axial),
                          'coronal':self.get_gradients_helper(self.gradients_coronal, self.gradient_names_coronal),
                          'sagittal':self.get_gradients_helper(self.gradients_sagittal, self.gradient_names_sagittal)}
        return all_view_dicts
    
    def get_gradients_helper(self, gradients, gradient_names):
        gradients_dict = {}
        for idx in range(len(gradient_names)):
            name = gradient_names[idx]
            grad = gradients[idx]
            gradients_dict[name] = grad
        return gradients_dict
    
    def extract_view_representation(self, x, view, conv_layers, fc_layer):
        if view == 'axial':
            save_gradient_func = self.save_gradient_axial
            gradient_names = self.gradient_names_axial
        elif view == 'coronal':
            save_gradient_func = self.save_gradient_coronal
            gradient_names = self.gradient_names_coronal
        elif view == 'sagittal':
            save_gradient_func = self.save_gradient_sagittal
            gradient_names = self.gradient_names_sagittal
        
        assert list(x.shape)==[15,3,420,420]
        
        #Iterate through first part of model: out shape [slices,512,14,14]
        for name, module in self.features:
            print('Applying features layer',name,'to data')
            x = module(x)
        
        #Iterate through second part of model, conv2d:
        #This is where the target activations and gradients come from.
        #out shape [slices, 16, 6, 6]
        target_activations = {}
        for name, module in conv_layers:
            print('Applying conv2d layer',name,'to data')
            x = module(x)
            print('\tdata shape after applying layer:',x.shape)
            if name in self.target_layers: #names are e.g. '4'. target_layers can be e.g. ['2','4']
                x.register_hook(save_gradient_func)
                gradient_names.append(name)
                target_activations[name] = x.cpu().data.numpy() #TODO is it okay to transfer to CPU here or do I have to wait? i.e. does it make a copy (in which case it is ok) or does it delete from gpu?
        
        x = fc_layer(x) #out shape [slices,83,1,1]
        x = torch.squeeze(x) #out shape [slices, 83]
        x_perslice_scores = x.transpose(0,1).unsqueeze(0) #out shape [1, 83, slices]. Scores for 83 diseases on every slice
        x = self.avgpool_1d(x_perslice_scores) #out shape [1, 83, 1]
        output = torch.squeeze(x, dim=2) #out shape [1, 83]
        return target_activations, x_perslice_scores, output
    
    def apply_chosen_comb_func(self,axial,coronal,sagittal):
        all_views = torch.cat((axial,coronal,sagittal),dim=0).transpose(0,1).unsqueeze(0) #out shape [1,83,3]
        #chosen_comb_func == 'avg':
        return self.comb_avg(all_views) #out shape [1,83,1]
       
    def run_model(self, x):
        """Run the model self.model on the input <x> and return the activations
        and output"""
        #Set up layers of the model to call
        self.features = self.model.features._modules.items()
        conv_axial = self.model.conv_axial._modules.items()
        conv_coronal = self.model.conv_coronal._modules.items()
        conv_sagittal = self.model.conv_sagittal._modules.items()
        fc_axial = self.model.fc_axial
        fc_coronal = self.model.fc_coronal
        fc_sagittal = self.model.fc_sagittal
        self.avgpool_1d = self.model.avgpool_1d
        self.comb_avg = self.model.comb_avg
        
        #Run model
        axial_activations, axial_perslice_scores, axial = self.extract_view_representation(x['axial'],'axial',conv_axial,fc_axial)
        coronal_activations, coronal_perslice_scores, coronal = self.extract_view_representation(x['coronal'],'coronal',conv_coronal,fc_coronal)
        sagittal_activations, sagittal_perslice_scores, sagittal = self.extract_view_representation(x['sagittal'],'sagittal',conv_sagittal,fc_sagittal)
        preds = self.apply_chosen_comb_func(axial,coronal,sagittal) #out shape [1,83,1]
        output = preds.squeeze(dim=2) #out shape [1,83]
        target_activations = {'axial':axial_activations,
                              'coronal':coronal_activations,
                              'sagittal':sagittal_activations}
        x_perslice_scores = {'axial':axial_perslice_scores,
                             'coronal':coronal_perslice_scores,
                             'sagittal':sagittal_perslice_scores}
        return target_activations, x_perslice_scores, output

class GradCamMultiview(gradcam.GradCam):
    def _make_figures(self, ctvol, volume_acc, target_layer_name,
                    disease_index, x_perslice_scores):
        """Save a gif showing the disease-specific heat map for the CT scan
        <ctvol> based on the target layer specified by <target_layer_name>
        Example <ctvol> shape: [15, 3, 420, 420]"""
        target_disease_name = self.disease_meanings[disease_index]
        
        for view in ['axial','coronal','sagittal']:
            ctvol_view = ctvol[view]
            #x_perslice_scores_this_disease shape [1, 15]
            #includes scores for this disease for each slice
            x_perslice_scores_this_disease = x_perslice_scores[view][:,disease_index,:]
            #slice_idx is an int, e.g. 13; the index of
            #the slice with the highest score for the selected disease
            slice_idx = (np.argsort(-x_perslice_scores_this_disease.cpu().data.numpy()).flatten()[0])
            
            print('Doing grad cam for',view,volume_acc,target_disease_name,'target layer',target_layer_name)
            anon_savename = target_disease_name+'_layer'+target_layer_name+'_slice'+str(slice_idx)+'_'+view
            savepath = os.path.join(self.results_dir, volume_acc.replace('.npz',anon_savename))
            
            #Select gradients and activations for the target layer:
            target_grads = self.grads_dict[view][target_layer_name].cpu().data.numpy() #e.g. out shape [15, 32, 10, 10]
            target_activs = self.target_activations[view][target_layer_name] #e.g. out shape [15, 32, 10, 10]
            
            #GIF of the entire CT scan
            gradcam.make_whole_gif(target_activs, target_grads, ctvol_view, savepath.replace('_slice'+str(slice_idx),''))
            
            #Make GIF and 2D plot for only the key slice indicated by <slice_idx>
            activ_slice, alpha_ks_slice, ctvol_slice = gradcam._select_slice(target_activs, target_grads, ctvol_view, slice_idx)
            gradcam.make_slice_gif(activ_slice, alpha_ks_slice, ctvol_slice, savepath)
            gradcam.make_slice_2d_debugging_plot(activ_slice, alpha_ks_slice, ctvol_slice, anon_savename, savepath)

#BodyAvgMultiview-sharedconv-avg conv2d layer numbers:
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
    gradcam.RunGradCam(descriptor='BodyAvgMultiview-sharedconv-avg',
                gradcamclass = GradCamMultiview,
                modeloutputsclass = ModelOutputs_BodyAvgMultiview,
                custom_net = custom_models_multiview.BodyAvgMultiview,
                custom_net_args = {'n_outputs':83,'chosen_conv':'sharedconv','chosen_comb_func':'avg'},
                params_path = '/home/rlb61/data/img-hiermodel2/results/2020-07-02_BodyAvgMultiview-sharedconv-avg/params/BodyAvgMultiview-sharedconv-avg',
                dataset_class = custom_datasets.CTDataset_2019_10,
                target_layer_names = ['6'], #e.g. ['3','4']
                dataset_args = {'label_type_ld':'location_disease_0323',
                                'view':'all', #VIEW IS ALL FOR A MULTIVIEW MODEL!!!
                                'projections9':True, 
                                'data_augment':{'train':True,
                                            'valid':False,#normally False, except for self-supervised learning
                                            'test':False},
                                'volume_prep_args':{
                                            'pixel_bounds':[-1000,200],
                                            'max_slices':45,
                                            'max_side_length':420,
                                            'num_channels':3,
                                            'crop_type':'single',
                                            'selfsupervised':False,
                                            'from_seg':False},
                                'selected_note_acc_files':{'train':'/home/rlb61/data/img-hiermodel2/load_dataset/ground_truth/predefined_subsets/2020-01-10-imgtrain_random2000.csv',
                                                           'valid':'/home/rlb61/data/img-hiermodel2/load_dataset/ground_truth/predefined_subsets/2020-01-10-imgvalid_a_random1000.csv'}
                                })