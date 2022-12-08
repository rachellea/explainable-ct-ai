#mask.py
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

import os
import copy
import timeit
import imageio
import numpy as np

import torch
import torch.nn as nn

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.ioff()

import warnings
warnings.filterwarnings('ignore')

from src.load_dataset.vol_proc import ctvol_preproc

####################################
# Construct Attention Ground Truth #--------------------------------------------
####################################
class ConstructAttnGroundTruth(object):
    def __init__(self, verbose, attn_storage_dir, sample, extrema, max_slices,
                 max_side_length, use_projections9, n_outputs, dilate,
                 downsamp_mode, segmap_path):
        """For the specified CT volume, return an attention ground truth which
        specifies the low-dimensional locations in which each of the n_outputs
        diseases is allowed to appear. The attention ground truth is a Tensor
        with shape [n_outputs, slices, 6, 6]
              (where slices = max_slices/3)
        Note that if the attention ground truth has already been created for this
        scan for this experiment (i.e. if we've already done epoch 0),
        then the attention ground truth will be loaded off disk rather than
        created again.
        The attention ground truth assumes the baseline axial orientation of
        the CT volume. The augmentation of the attention ground truth (with
        flips and rotations to match those applied to the CT volume) occurs
        in losses.py because this augmentation is likely to be different
        every epoch.
        
        Variables:
        <verbose>: if True then print off progress and also save visualizations
            of the masks.
        <attn_storage_dir>: the directory in which to save the attention ground
            truth
        <sample> is a dict that includes keys 'volume_acc'
            (the volume accession number) and 'auglabel' (a vector
            describing the data transformations that were applied)
        <extrema> is the extrema file described in ctvol_preproc.py
        <max_slices>: int for the max number of slices in the processed CT scan
        <max_side_length>: int for the max side length in the processed CT scan
        <use_projections9>: True if precomputed 9-projections are being used,
            False otherwise
        <n_outputs>: int; total number of abnormalities that will be
            predicted
        <dilate>: if True then perform a dilation operation on the small
            organ masks in order to include a little more area around
            each organ. (Motivation: to help the performance of pleural effusion
            which is located outside the lungs, and to help the performance
            of findings like CABG which may extend outside the heart e.g.
            into the sternum)
        <downsamp_mode>: string specifying the algorithm used to downsample
            the organ segmentation masks. Corresponds to the options available
            for the 'mode' for the function nn.functional.interpolate, for
            5D data: 'nearest' | 'trilinear' | 'area'
            ('linear' is for 3D only , 'bilinear' is for 4D only, and
            'bicubic' is for 4D only so those are not options here.)
        <segmap_path> is the path to the directory containing the
            binary precomputed lung segmentation masks
            e.g. /storage/rlb61-ct_images/vna/rlb61/2019-10-BigData-segmaps-DEID/"""
        self.verbose = verbose
        self.attn_storage_dir = attn_storage_dir
        self.sample = sample
        self.extrema = extrema
        self.max_slices = max_slices
        self.max_side_length = max_side_length
        self.use_projections9 = use_projections9
        self.n_outputs = n_outputs
        self.dilate = dilate
        self.downsamp_mode = downsamp_mode
        assert self.downsamp_mode in ['nearest','trilinear','area']
        self.segmap_path = segmap_path
        if self.verbose: print('dilate=',self.dilate,
                               '\ndownsamp_mode=',self.downsamp_mode)
        
        #Get the path to this specific example. If the attn_gr_truth already
        #exists then load it. If it doesn't exist, create it and save it.
        volume_acc_no_ext = sample['volume_acc'].replace('.npz','').replace('.npy','')
        self.attn_gr_truth_file_path = os.path.join(attn_storage_dir, volume_acc_no_ext+'_attn_gr_truth.pt')
        if os.path.isfile(self.attn_gr_truth_file_path):
            if self.verbose: print('Loading attn_gr_truth from existing file')
            self.attn_gr_truth = torch.load(self.attn_gr_truth_file_path)
        else:
            if self.verbose: print('Creating attn_gr_truth from scratch')
            self.attn_gr_truth = self.create_and_save_attn_gr_truth()
            
    def create_and_save_attn_gr_truth(self):
        tot0 = timeit.default_timer()
        #Get the overall mask which indicates the segmentation for all the organs
        full_size_mask_np = GetMask(self.verbose, self.attn_storage_dir, self.sample, self.extrema,
                            self.max_slices, self.max_side_length,
                            self.use_projections9, self.segmap_path).return_mask()
        
        #Convert to Tensor
        full_size_mask = torch.Tensor(full_size_mask_np)
        
        #Get the downsampled organ masks
        #slices is the number of slices in the low-dimensional representation,
        #which is max_slices divided by three because to run the network
        #we make the 1-channel volume into a 3-channel volume.
        slices = int(self.max_slices/3)
        if int(full_size_mask.sum())==0:
            #the mask is only zero if the segmentation was bad. So if the segmentation
            #was bad, create your own rough heart, right_lung, and left_lung masks
            #by selecting the middle half, right half, or left half of the scan
            #respectively.
            heart_seg_small, right_lung_seg_small, left_lung_seg_small = create_rough_seg_masks_from_scratch(slices)
        else:
            #If we have a good quality mask, then we need to use that mask to
            #calculate constructed_attn_gr_truth based on the classification
            #gr truth and the organ segmentation masks. 
            #Each of these small organ masks has shape e.g. [15,6,6]
            heart_seg_small, right_lung_seg_small, left_lung_seg_small = shrink_the_seg_masks(full_size_mask, slices, self.downsamp_mode)
        
        #Delete the full size mask to avoid excess memory usage
        del full_size_mask
        
        #Dilate the organ masks if applicable
        if self.dilate:
            right_lung_seg_small = dilate_small_mask(right_lung_seg_small)
            heart_seg_small = dilate_small_mask(heart_seg_small)
            left_lung_seg_small = dilate_small_mask(left_lung_seg_small)
        
        #Sanity check visualization
        if self.verbose:
            #Upsample the masks again in order to do the visualization:
            organ_masks = {
                'right_lung':nn.functional.interpolate(right_lung_seg_small.unsqueeze(0).unsqueeze(0), size=[self.max_slices,self.max_side_length,self.max_side_length], mode=self.downsamp_mode).squeeze(),
                'heart':nn.functional.interpolate(heart_seg_small.unsqueeze(0).unsqueeze(0), size=[self.max_slices,self.max_side_length,self.max_side_length], mode=self.downsamp_mode).squeeze(),
                'left_lung':nn.functional.interpolate(left_lung_seg_small.unsqueeze(0).unsqueeze(0), size=[self.max_slices,self.max_side_length,self.max_side_length], mode=self.downsamp_mode).squeeze()}
            make_gifs_of_mask('after_downsamp_dil'+str(self.dilate)+'_'+self.downsamp_mode, self.sample, organ_masks, self.attn_storage_dir, mask_orientation='axial')
            #TODO maybe make visualizations that all use trilinear interpolation for upsampling.
        
        #Construct the final attention ground truth
        attn_gr_truth = construct_attn_gr_truth(self.sample, heart_seg_small, left_lung_seg_small, right_lung_seg_small)
        
        #Save it
        torch.save(attn_gr_truth, self.attn_gr_truth_file_path)
        tot1 = timeit.default_timer()
        if self.verbose: print('Total Attn Gr Truth Time', round((tot1 - tot0),2),'seconds')
        return attn_gr_truth

def create_rough_seg_masks_from_scratch(slices): #Done with testing
    """This function is only called when we have a low quality segmentation
    mask that is not usable, e.g. because a lung is missing. In that case,
    we need to make rough masks.
    Output shape of [slices,6,6] per mask is hardcoded into this function."""
    slices = int(slices)
    
    #Make the allowed heart region the middle 2/3
    heart_seg_small = np.zeros((slices,6,6))
    heart_seg_small[:,:,1:5] = 1 #indices 1, 2, 3, 4
    
    #Make the allowed right lung region the right half. Note that right is the axis2min.
    right_lung_seg_small = np.zeros((slices,6,6))
    right_lung_seg_small[:,:,0:3] = 1 #indices 0, 1, 2
    
    #Make the allowed left lung region the left half. Note that left is the axis2max
    left_lung_seg_small = np.zeros((slices,6,6))
    left_lung_seg_small[:,:,3:] = 1 #indices 3, 4, 5
    
    #Now convert to float Tensor and return. Must make the arrays contiguous
    #otherwise there will be a negative strides error.
    heart_seg_small = torch.Tensor(np.ascontiguousarray(heart_seg_small)).float()
    right_lung_seg_small = torch.Tensor(np.ascontiguousarray(right_lung_seg_small)).float()
    left_lung_seg_small = torch.Tensor(np.ascontiguousarray(left_lung_seg_small)).float()
    return heart_seg_small, right_lung_seg_small, left_lung_seg_small

def shrink_the_seg_masks(full_size_mask, slices, downsamp_mode): #Done with testing
    """Helper function which will extract organ-specific binary masks and then
    shrink them down to the necessary size.
    Output shape of [slices,6,6] per mask is hardcoded into this function."""
    slices = int(slices)
    
    #Must convert to float because the next step, interpolation,
    #requires a float data type and by default the batch['mask]==n step
    #will produce a torch.uint8 data type
    heart_seg = (full_size_mask==2).float() #out shape of binary mask [1,bigslices,bigsquare,bigsquare] e.g. [1,45,420,420]
    right_lung_seg = (full_size_mask==1).float()
    left_lung_seg = (full_size_mask==3).float()
    
    #Each of the full sized binary masks could be for example [45,420,420]
    #and we need to get it down to [slices,6,6]. 
    #The function nn.functional.interpolate assumes there are 5 dimensions
    #for 3D interpolation, so we unsqueeze the input to get e.g.
    #shape [1,1,max_slices,420,420] and then after we've done the interpolation we
    #squeeze it to get [slices,6,6] as our final output shape.
    #max_slices could be 45 (when there are 9-projections) in which case
    #slices would be 45/3 = 15 (after doing 3 channels we get 15)
    heart_seg_small = nn.functional.interpolate(heart_seg.unsqueeze(0).unsqueeze(0), size=(slices,6,6), mode=downsamp_mode).squeeze()
    right_lung_seg_small = nn.functional.interpolate(right_lung_seg.unsqueeze(0).unsqueeze(0), size=(slices,6,6), mode=downsamp_mode).squeeze()
    left_lung_seg_small = nn.functional.interpolate(left_lung_seg.unsqueeze(0).unsqueeze(0), size=(slices,6,6), mode=downsamp_mode).squeeze()
    
    #Anything that is not zero, round up to 1. This is because when we do the
    #loss function, we forbid the model from looking in any areas that are
    #equal to zero - so effectively all elements which are nonzero are treated
    #as allowed, i.e. as a 'one'
    heart_seg_small = (heart_seg_small!=0).float()
    right_lung_seg_small = (right_lung_seg_small!=0).float()
    left_lung_seg_small = (left_lung_seg_small!=0).float()
    
    return heart_seg_small, right_lung_seg_small, left_lung_seg_small

def dilate_small_mask(small_mask): #Done with testing
    """Morphological dilation of the small mask. Motivation: if we expand the
    lung masks in particular then we will include the pleura which will help
    pleural effusion performance"""
    #See https://stackoverflow.com/questions/56235733/is-there-a-tensor-operation-or-function-in-pytorch-that-works-like-cv2-dilate
    kernel = torch.ones(1,1,3,3,3).float()
    #Add padding before the dilation so that the output will be of the same
    #shape as the input, e.g. [15,6,6]
    small_mask_padded = torch.nn.functional.pad(small_mask, [1,1,1,1,1,1], mode='constant', value=0)
    #Perform the dilation operation:
    small_mask_dilated = torch.clamp(torch.nn.functional.conv3d(small_mask_padded.unsqueeze(0).unsqueeze(0), kernel, stride=(1,1,1), padding=(0,0,0)), 0, 1).squeeze()
    return small_mask_dilated

def construct_attn_gr_truth(sample, heart_seg_small, left_lung_seg_small,
                            right_lung_seg_small): #Done with testing
    """Helper function that returns a disease-specific volumetric attention
    ground truth with shape [n_outputs, slices, 6, 6] where n_outputs is the
    number of diseases and slices x 6 x 6 is the 3D low-dimensional
    representation of the CT scan."""
    heart_gr_truth = sample['heart_gr_truth']
    left_lung_gr_truth = sample['left_lung_gr_truth']
    right_lung_gr_truth = sample['right_lung_gr_truth']
    
    n_outputs_heart = heart_gr_truth.shape[0] #e.g. heart_gr_truth has shape [30] so n_outputs_heart = 30
    n_outputs_lung = left_lung_gr_truth.shape[0] #e.g. left_lung_gr_truth has shape [51] so n_outputs_lung = 51
    n_outputs = n_outputs_heart + n_outputs_lung #e.g. n_outputs = 81
    
    #constructed_attn_gr_truth shape [n_outputs, slices, 6, 6] e.g. [81, 15, 6, 6]
    #We want to construct a ground truth for what regions are allowed to be
    #focused on for each disease. If the disease is absent, leave the constructed
    #ground truth as zero (saying the attention is not allowed to be active).
    #If the disease is present, say that the attention has to be within
    #the organ(s) in which the disease occurs.
    constructed_attn_gr_truth = torch.zeros(n_outputs, heart_seg_small.shape[0], heart_seg_small.shape[1], heart_seg_small.shape[2]).float()
    for heart_dz_idx in range(0,n_outputs_heart):
        if heart_gr_truth[heart_dz_idx] == 1:
            constructed_attn_gr_truth[heart_dz_idx,:,:,:] = heart_seg_small
    for lung_dz_idx in range(0,n_outputs_lung):
        if left_lung_gr_truth[lung_dz_idx] == 1:
            constructed_attn_gr_truth[lung_dz_idx+n_outputs_heart,:,:,:] += left_lung_seg_small
        #Super important that this next line is a separate 'if' and NOT an 'elif'!
        #'elif' is WRONG! It is perfectly acceptable to have the disease present
        #in both lungs!
        if right_lung_gr_truth[lung_dz_idx] == 1:
            constructed_attn_gr_truth[lung_dz_idx+n_outputs_heart,:,:,:] += right_lung_seg_small
    
    #In some cases, if there was a dilation operation applied to the lung
    #masks, it's possible that the left_lung_seg_small and right_lung_seg_small
    #could overlap. To keep everything as clean as possible, let's make sure
    #it's a binary tensor:
    constructed_attn_gr_truth = (constructed_attn_gr_truth!=0).float()
    return constructed_attn_gr_truth

#######################
# Get Full Sized Mask #---------------------------------------------------------
#######################
class GetMask(object):
    def __init__(self, verbose, attn_storage_dir, sample, extrema, max_slices,
                 max_side_length, use_projections9, segmap_path):
        """For the specified CT volume, return a mask which is either:
            * a 3D np array whose elements are 0 for background, 1 for the
              right lung, 2 for the heart, and 3 for the left lung.
            * OR in the case where the segmentation mask is messed up, the
              mask will be a dummy mask, a 3D np array that contains only zeros."""
        self.verbose = verbose #if verbose is True then sanity check GIFs are produced
        self.attn_storage_dir = attn_storage_dir
        self.sample = sample
        self.extrema = extrema
        self.max_slices = max_slices
        self.max_side_length = max_side_length
        self.use_projections9 = use_projections9
        self.segmap_path = segmap_path
        self.volume_acc = sample['volume_acc']
        self.auglabel = sample['auglabel']
        self.randpad6val = sample['randpad6val']

    def return_mask(self):
        """Return the final mask."""
        tot0 = timeit.default_timer()
        outmask = self.get_raw_mask()
        if int(outmask.sum())!=0: #i.e. if it's an actual mask not a dummy mask:
            outmask = self.pad_and_crop_mask(outmask)
            outmask = outmask.copy() #copy so that numpy array will not have any negative strides
            if self.verbose:
                tot1 = timeit.default_timer()
                print('Total Full Mask Time', round((tot1 - tot0),2),'seconds')
                self.make_gifs_of_full_size_mask(outmask)
        return outmask
    
    def get_raw_mask(self):
        """Return a 3D numpy array whose elements are 0 for background, 1 for right
        lung, 2 for heart, and 3 for left lung. The array is projected across
        9 slices if self.use_projections9 is True."""
        #Load raw mask and split it into organs
        raw_mask = load_raw_mask_and_split_it_into_organs(self.volume_acc, self.segmap_path, self.extrema)
        
        #Apply projections if applicable
        if self.use_projections9:
            raw_mask = project_volume_axial(raw_mask)
        return raw_mask
    
    def pad_and_crop_mask(self, raw_mask):
        """Crop and pad mask in the same way that the CT volume was cropped
        and padded."""
        #Padding to minimum size
        pad_mask = ctvol_preproc.pad_volume(raw_mask, self.max_slices, self.max_side_length)
        
        #Random padding that matches the random padding that was used on the ctvol
        assert np.sum(self.randpad6val)==0 #check inserted on 9/22/2020 since I am
        #no longer doing random padding as a form of data augmentation.
        #Why? Because random padding has minimal effect on the performance, and
        #it prevents me from computing the attention ground truth only once.
        #In the future, if I bring back random padding, uncomment the following:
        #pad_mask = np.pad(pad_mask, pad_width = ((self.randpad6val[0],self.randpad6val[1]),
        #    (self.randpad6val[2],self.randpad6val[3]),
        #    (self.randpad6val[4], self.randpad6val[5])),
        #     mode = 'constant', constant_values = 0)
        
        #Crop
        pad_mask = ctvol_preproc.single_crop_3d_fixed(pad_mask, self.max_slices, self.max_side_length)
        return pad_mask
    
    def make_gifs_of_full_size_mask(self, final_mask):
        figs = {'right_lung':1,'heart':2,'left_lung':3}
        organ_masks = {}
        for organname in figs.keys():
            organvalue = figs[organname]
            organ_masks[organname] = (final_mask == organvalue).astype('uint8')
        make_gifs_of_mask('highres_orig_mask',self.sample, organ_masks, self.attn_storage_dir,
                          mask_orientation = 'axial')

def load_raw_mask_and_split_it_into_organs(volume_acc, segmap_path, extrema):
    """Return a 3D numpy array whose elements are 0 for background, 1 for right
    lung, 2 for heart, and 3 for left lung. If the segmentation is bad quality
    according to the right border or left border (i.e. if the right lung or left
    lung is missing) then return a 3D numpy array whose elements are all 0 (all
    background).
    
    <volume_acc> is the volume accession specifying the CT scan, e.g. RHAA12345_5.npz
    <segmap_path>: path to the directory where the segmentation mask
        of both lungs is stored.
    <extrema> is the extrema file specifying the overall bounding box that
        surrounds both lungs.
    
    This function is used in GetMask() and find_separate_organ_bboxes.py"""
    #Raw mask
    raw_mask = np.load(os.path.join(segmap_path, volume_acc.replace('.npz','')+'_seg.npz'))['segmap']
    
    #Raw lung bbox from extrema file
    sup_axis0min = extrema.at[volume_acc,'sup_axis0min']
    inf_axis0max = extrema.at[volume_acc,'inf_axis0max']
    ant_axis1min = extrema.at[volume_acc,'ant_axis1min']
    pos_axis1max = extrema.at[volume_acc,'pos_axis1max']
    rig_axis2min = extrema.at[volume_acc,'rig_axis2min']
    lef_axis2max = extrema.at[volume_acc,'lef_axis2max']
    
    #Change the heart values to 2
    #Do this first, because in this step, the lungs are both assumed to
    #be filled with ones, but in a later step we want to make the left
    #lung be filled with 3s.
    #Assume that the heart is the stuff that is equal to zero within the
    #lung bounding box (i.e. the non-lung stuff within the lung bbox)
    #In the lung_bbox_crop, lung=1 and nonlung=0
    #Make this bbox narrower in the right-left direction to focus on the
    #center where the heart is, and skew it towards the left since
    #the heart is on the left:
    quarter_distance = int((lef_axis2max-rig_axis2min)/4.0)
    lung_bbox_crop = raw_mask[sup_axis0min:inf_axis0max,ant_axis1min:pos_axis1max,rig_axis2min+quarter_distance:lef_axis2max-int(0.5*quarter_distance)]
    #In the heart, heart=2 and nonheart=0
    #First, call everything that is non-lung the 'heart':
    heart = ((lung_bbox_crop==0).astype('int'))*2
    #in heart_and_lungs, lung=1 and heart=2
    heart_and_lungs = lung_bbox_crop+heart
    raw_mask[sup_axis0min:inf_axis0max,ant_axis1min:pos_axis1max,rig_axis2min+quarter_distance:lef_axis2max-int(0.5*quarter_distance)] = heart_and_lungs

    #Change the left_lung values to 3, assuming that if you bisect
    #the volume in the center you will divide the lungs into right vs left
    right_left_centerline = int(raw_mask.shape[2]/2.0)
    left_lung = raw_mask[:,:,right_left_centerline:]
    left_lung[left_lung==1] = 3
    raw_mask[:,:,right_left_centerline:] = left_lung
    
    #Get the 'fixed' values (if applicable) for the lung boundaries, so you
    #can detect whether the lungs are both present or not:
    fixed_rig_axis2min = ctvol_preproc.fix_right_border(rig_axis2min, extrema)
    fixed_lef_axis2max = ctvol_preproc.fix_left_border(lef_axis2max, extrema)
    
    #If right border or left border are messed up (e.g. because a lung
    #is missing) then record that the segmentation is
    #messed up by saving a Tensor containing all zeros.
    #A messed up right border has a value that is too big. The
    #fixed value will be smaller (a smaller min). 
    fixed_rig_axis2min = ctvol_preproc.fix_right_border(rig_axis2min, extrema)
    #A messed up left border has a value that is too small. The fixed
    #value will be bigger (a bigger max)
    fixed_lef_axis2max = ctvol_preproc.fix_left_border(lef_axis2max, extrema)
    if ((fixed_rig_axis2min < rig_axis2min) or (fixed_lef_axis2max > lef_axis2max)):
        raw_mask = np.zeros((raw_mask.shape[0],raw_mask.shape[1],raw_mask.shape[2]))
    
    return raw_mask

####################################################
# Visualization Function for Sanity Check Purposes #----------------------------
####################################################
def make_gifs_of_mask(description, sample, organ_masks, attn_storage_dir,
                      mask_orientation):
    """Save GIFs of the CT volume showing the masks of different organs.
    
    <sample> is a sample produced by custom_datasets.py. Note that the CT
        volume in this sample has been transformed with a flip and/or a
        rotation. The sample has NOT been created with a DataLoader, so there
        is NO batch dimension.
    <organ_masks> is a dictionary with keys 'right_lung','heart','left_lung' and
        values that are organ masks for those organs.
        Each organ mask is either a 3D numpy array or a 3D torch Tensor.
        Each organ mask has the same shape as <ctvol_tensor>
        (Exception: in unit_tests_mask.py <organ_masks> is actually used for
        disease masks, but it has the same format of string keys and Tensor
        values which are masks.)
    <mask_orientation>: either 'axial' or 'transformed.' If 'axial' then it
        can be used as-is. If 'transformed' then it needs to have the
        transformation reversed at the same time that the CT volume's
        transformation is reversed."""
    assert mask_orientation in ['axial','transformed']
    
    #Turn CT volume into numpy array
    ctvol = copy.deepcopy(sample['data'].cpu().numpy())
    
    #Turn organ masks into numpy arrays if they are Tensors:
    for organname in list(organ_masks.keys()):
        if isinstance(organ_masks[organname],torch.Tensor):
            organ_masks[organname] = organ_masks[organname].cpu().numpy()
    
    #if a three-channel image, reshape back to 1 channel
    if ctvol.shape[1]==3:
        ctvol = np.reshape(ctvol, newshape=[ctvol.shape[0]*ctvol.shape[1], ctvol.shape[2], ctvol.shape[3]])
    
    #Note that the ctvol's values have already been normalized tp 0 -1
    #Scale them to the range 0 to 255 and cast to uint8
    ctvol = (ctvol*255).astype('uint8')
    
    #Undo the transformations to put the ctvol back into axial orientation,
    #because the ctvol is always transformed.
    rot_vector = np.array([int(x) for x in sample['auglabel'][3:].tolist()])
    flip_vector = np.array([int(x) for x in sample['auglabel'][0:3].tolist()])
    ctvol = undo_flip_and_rotation(ctvol,rot_vector,flip_vector)
    #Also, if <mask_orientation> is 'transformed' then also put the masks
    #back into axial orientation:
    if mask_orientation == 'transformed':
        for organname in list(organ_masks.keys()):
            organ_masks[organname] = undo_flip_and_rotation(organ_masks[organname],rot_vector,flip_vector)
    
    #Make gifs
    volume_acc = sample['volume_acc'].replace('.npy','').replace('.npz','')
    norm_ctvol = matplotlib.colors.Normalize(vmin=np.amin(ctvol),vmax=np.amax(ctvol),clip=False)
    #make a norm for the heatmap too because we want heatmaps that are all ones
    #to appear as all red, not all purple:
    norm_heatmap = matplotlib.colors.Normalize(vmin=0,vmax=1,clip=False)
    for organname in list(organ_masks.keys()):
        images = []
        organ_mask = organ_masks[organname]
        for slicenum in range(ctvol.shape[0]):
            fig, ax = plt.subplots(figsize=(8,6))
            #show CT scan in grayscale
            plt.imshow(ctvol[slicenum,:,:],cmap='gray',norm=norm_ctvol)
            #Show the organ mask in a color
            slice_organ_mask = organ_mask[slicenum,:,:]
            plt.imshow(slice_organ_mask, cmap='rainbow', alpha=0.3, norm=norm_heatmap)
            plt.title(organname)
            fig.canvas.draw() #draw the canvas, cache the renderer
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            images.append(image)
            plt.close()
        savepath = os.path.join(attn_storage_dir, volume_acc.replace('.npz','').replace('.npy','')+description+organname+'.gif')
        imageio.mimsave(savepath,images,duration=0.02)

def undo_flip_and_rotation(volume,rot_vector,flip_vector):
    """Return <volume> for which the flip specified by <flip_vector> has been
    undone, and the rotation specified by <rot_vector> has been undone"""
    if 1 in rot_vector:
        chosen_k = np.where(rot_vector==1)[0][0]+1 #indices are 0, 1, 2 for k equal to 1, 2, or 3 respectively
        reverse_k = 4-chosen_k
        volume = np.rot90(volume, k=reverse_k, axes=(1,2))
    if 1 in flip_vector:
        chosen_axis = np.where(flip_vector==1)[0][0]
        volume =  np.flip(volume, axis=chosen_axis)
    return volume

def project_volume_axial(volume):
    n_pool = 9
    pooling = nn.MaxPool3d(kernel_size = (n_pool,1,1), stride=(n_pool,1,1), padding=0)
    temp = torch.Tensor(volume)
    temp = torch.unsqueeze(temp,dim=0)
    temp = pooling(temp)
    temp = torch.squeeze(temp)
    return temp.numpy()