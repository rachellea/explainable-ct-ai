#extract_disease_reps.py
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
import numpy as np

def return_segprediction_from_disease_rep(out, chosen_label_index):
    """Run the provided <model> on the provided <ctvol> and return the predicted
    segmentation for the disease specified by <chosen_label_index>"""    
    disease_reps = out['disease_reps'] #out shape [80, 15, 16, 6, 6]
    #Now sum over the feature dimension. This the same as your 'improved'
    #Grad-CAM where you first element-wise multiply the entire representation against
    #the entire gradients (in this case, literally, the entire set of weights), before
    #taking an average. (Different from vanilla Grad-CAM where you first average the
    #gradients and then multiply each feature map by the resulting scalar.)
    #Note that we actually sum rather than average because in losses.py we
    #sum rather than average. Summing vs averaging shouldn't change anything in
    #the final normalized map.
    disease_reps_collapsed_features = torch.sum(disease_reps,dim=2).data.cpu().numpy() #out shape [80, 15, 6, 6]
    
    #Extract the segprediction:
    segprediction = disease_reps_collapsed_features[chosen_label_index,:,:,:] #out shape [15, 6, 6]
    
    #Figure out the slice_idx of the slice with the highest score for the
    #chosen disease:
    x_perslice_scores_this_disease = out['x_perslice_scores'].cpu().data.numpy()[:,chosen_label_index,:]
    return segprediction, x_perslice_scores_this_disease         