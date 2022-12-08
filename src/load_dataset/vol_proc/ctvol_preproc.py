#ctvol_preproc.py
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
import pandas as pd

"""CT volume preprocessing functions"""

#############################################
# Pixel Values (on torch Tensors for speed) #-----------------------------------
#############################################
def normalize(ctvol, lower_bound, upper_bound): #Done testing
    """Clip images and normalize"""
    #formula https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range
    ctvol = torch.clamp(ctvol, lower_bound, upper_bound)
    ctvol = (ctvol - lower_bound) / (upper_bound - lower_bound)
    return ctvol

def torchify_pixelnorm_pixelcenter(ctvol, pixel_bounds):
    """Normalize using specified pixel_bounds and then center on the ImageNet
    mean. Used in 2019_10 dataset preparation"""
    #Cast to torch Tensor
    #use torch Tensor instead of numpy array because addition, subtraction,
    #multiplication, and division are faster in torch Tensors than np arrays
    ctvol = torch.from_numpy(ctvol).type(torch.float)
    
    #Clip Hounsfield units and normalize pixel values
    ctvol = normalize(ctvol, pixel_bounds[0], pixel_bounds[1])
    
    #Center on the ImageNet mean since you are using an ImageNet pretrained
    #feature extractor:
    ctvol = ctvol - 0.449
    return ctvol

###########
# Padding #---------------------------------------------------------------------
###########
def pad_slices(ctvol, max_slices): #Done testing
    """For <ctvol> of shape (slices, side, side) pad the slices to shape
    max_slices for output of shape (max_slices, side, side)"""
    padding_needed = max_slices - ctvol.shape[0]
    assert (padding_needed >= 0), 'Image slices exceed max_slices by'+str(-1*padding_needed)
    if padding_needed > 0:
        before_padding = int(padding_needed/2.0)
        after_padding = padding_needed - before_padding
        ctvol = np.pad(ctvol, pad_width = ((before_padding, after_padding), (0,0), (0,0)),
                     mode = 'constant', constant_values = np.amin(ctvol))
        assert ctvol.shape[0]==max_slices
    return ctvol

def pad_sides(ctvol, max_side_length): #Done testing
    """For <ctvol> of shape (slices, side, side) pad the sides to shape
    max_side_length for output of shape (slices, max_side_length,
    max_side_length)"""
    needed_padding = 0
    for side in [1,2]:
        padding_needed = max_side_length - ctvol.shape[side]
        if padding_needed > 0:
            before_padding = int(padding_needed/2.0)
            after_padding = padding_needed - before_padding
            if side == 1:
                ctvol = np.pad(ctvol, pad_width = ((0,0), (before_padding, after_padding), (0,0)),
                         mode = 'constant', constant_values = np.amin(ctvol))
                needed_padding += 1
            elif side == 2:
                ctvol = np.pad(ctvol, pad_width = ((0,0), (0,0), (before_padding, after_padding)),
                         mode = 'constant', constant_values = np.amin(ctvol))
                needed_padding += 1
    if needed_padding == 2: #if both sides needed to be padded, then they
        #should be equal (but it's possible one side or both were too large
        #in which case we wouldn't expect them to be equal)
        assert ctvol.shape[1]==ctvol.shape[2]==max_side_length
    return ctvol

def pad_volume(ctvol, max_slices, max_side_length):
    """Pad <ctvol> to a minimum size of
    [max_slices, max_side_length, max_side_length], e.g. [402, 308, 308]
    Used in 2019_10 dataset preparation"""
    if ctvol.shape[0] < max_slices:
        ctvol = pad_slices(ctvol, max_slices)
    #Note: for the dataset that is produced based on the lung bbox from the lung
    #segmentation (i.e. when from_seg = True) it is REALLY important that we
    #have a check for when ctvol.shape[1] < max_side_length and a SEPARATE
    #check for when ctvol.shape[2] < max_side_length because it's possible that
    #shape[1] != shape[2]. If we only had a check for shape[1] then when this
    #function is given a CT scan of shape [max_slices, max_side_length,
    #something_else] it would fail to pad something_else to max_side_length. 
    if ctvol.shape[1] < max_side_length:
        ctvol = pad_sides(ctvol, max_side_length)
    if ctvol.shape[2] < max_side_length:
        ctvol = pad_sides(ctvol, max_side_length)
    return ctvol

###########################
# Reshaping to 3 Channels #-----------------------------------------------------
###########################
def sliceify(ctvol): #Done testing
    """Given a numpy array <ctvol> with shape [slices, square, square]
    reshape to 'RGB' [max_slices/3, 3, square, square]"""
    return np.reshape(ctvol, newshape=[int(ctvol.shape[0]/3), 3, ctvol.shape[1], ctvol.shape[2]])

def reshape_3_channels(ctvol):
    """Reshape grayscale <ctvol> to a 3-channel image
    Used in 2019_10 dataset preparation"""
    if ctvol.shape[0]%3 == 0:
        ctvol = sliceify(ctvol)
    else:
        if (ctvol.shape[0]-1)%3 == 0:
            ctvol = sliceify(ctvol[:-1,:,:])
        elif (ctvol.shape[0]-2)%3 == 0:
            ctvol = sliceify(ctvol[:-2,:,:])
    return ctvol

##################################
# Cropping and Data Augmentation #----------------------------------------------
##################################
def crop_specified_axis(ctvol, max_dim, axis): #Done testing
    """Crop 3D volume <ctvol> to <max_dim> along <axis>"""
    dim = ctvol.shape[axis]
    if dim > max_dim:
        amount_to_crop = dim - max_dim
        part_one = int(amount_to_crop/2.0)
        part_two = dim - (amount_to_crop - part_one)
        if axis == 0:
            return ctvol[part_one:part_two, :, :]
        elif axis == 1:
            return ctvol[:, part_one:part_two, :]
        elif axis == 2:
            return ctvol[:, :, part_one:part_two]
    else:
        return ctvol

def single_crop_3d_fixed(ctvol, max_slices, max_side_length):
    """Crop a single 3D volume to shape [max_slices, max_side_length,
    max_side_length]"""
    ctvol = crop_specified_axis(ctvol, max_slices, 0)
    ctvol = crop_specified_axis(ctvol, max_side_length, 1)
    ctvol = crop_specified_axis(ctvol, max_side_length, 2)
    return ctvol

def single_crop_3d_fixed_from_seg(ctvol, max_slices, max_side_length,
                                  volume_acc, extrema): #Todo test this
    """Crop a single 3D volume to shape [max_slices, max_side_length,
    max_side_length] such that the crop is centered on the lung bounding
    box coordinates available in <extrema>"""
    sup_axis0min, inf_axis0max, ant_axis1min, pos_axis1max, rig_axis2min, lef_axis2max = get_clean_extrema_for_ctvol(ctvol, max_slices, max_side_length, volume_acc, extrema)
    
    #Perform crop
    ctvol = ctvol[sup_axis0min:inf_axis0max,
                  ant_axis1min:pos_axis1max,
                  rig_axis2min:lef_axis2max]
    
    #It's possible that a really small CT scan might be smaller than the needed
    #shape. In that case we need to pad. (Note that we pad after we make use of
    #the bbox coordinates because if we padded first, it would mess up
    #the meaning of the bbox coordinates)
    ctvol = pad_volume(ctvol, max_slices, max_side_length)
    
    error_end = ' instead of ['+str(max_slices)+','+str(max_side_length)+','+str(max_side_length)+']'
    assert ctvol.shape[0] == max_slices, 'Error: '+volume_acc+' processed shape is '+str(ctvol.shape)+error_end
    assert ctvol.shape[1] == max_side_length, 'Error: '+volume_acc+' processed shape is '+str(ctvol.shape)+error_end
    assert ctvol.shape[2] == max_side_length, 'Error: '+volume_acc+' processed shape is '+str(ctvol.shape)+error_end
    return ctvol

def single_crop_3d_augment(ctvol, max_slices, max_side_length, from_seg,
                           volume_acc, extrema):
    """Crop a single 3D volume to shape [max_slices, max_side_length,
    max_side_length] with randomness in the centering and random
    flips or rotations.
    If <from_seg> is True then the crop is approximately centered on the
    lung bounding box coordinates available in <extrema>"""
    #Introduce random padding so that the centered crop will be slightly random
    ctvol, randpad6val = rand_pad(ctvol)
    
    #Obtain the center crop
    if not from_seg:
        ctvol = single_crop_3d_fixed(ctvol, max_slices, max_side_length)
    else: #seg
        ctvol = single_crop_3d_fixed_from_seg(ctvol, max_slices, max_side_length,
                                          volume_acc, extrema)
    
    #Flip and rotate
    ctvol, flip_vector = rand_flip(ctvol)
    ctvol, rot_vector = rand_rotate(ctvol)
    
    #Make contiguous array to avoid Pytorch error
    return np.ascontiguousarray(ctvol), flip_vector, rot_vector, randpad6val

def triple_crop_3d_fixed(ctvol, max_slices, max_side_length, from_seg,
                           volume_acc, extrema):
    """Crop a single 3D volume into three sub-volumes: one for the right lung,
    one for the heart and mediastinum, and another for the left lung.
    If <from_seg> is True then the initial crop is approximately centered on the
    lung bounding box coordinates available in <extrema>"""
    #Obtain the center crop. Example cropped shape: [402,308,308]
    if not from_seg:
        ctvol = single_crop_3d_fixed(ctvol, max_slices, max_side_length)
    else: #seg
        ctvol = single_crop_3d_fixed_from_seg(ctvol, max_slices, max_side_length,
                                  volume_acc, extrema)
    
    #Crop out the pieces: right lung, heart, and left lung
    whole = ctvol.shape[1]
    half = int(round(ctvol.shape[1]/2)) #e.g. 308/2 = 154
    quarter = int(round(ctvol.shape[1]/4)) #e.g. 308/4 = 77
    ctvol_dict = {'right_lung':ctvol[:,:,0:half], #e.g. ctvol[:,:,0:154]
                  'heart':ctvol[:,:,quarter:(whole-quarter)], #e.g. ctvol[:,:,77:231]
                  'left_lung':ctvol[:,:,half:]} #e.g. ctvol[:,:,154:]
    return ctvol_dict

def triple_crop_3d_augment(ctvol, max_slices, max_side_length, from_seg,
                           volume_acc, extrema):
    """"Same as triple_crop_3d_fixed() except also includes data augmentation"""
    #Introduce random padding so that the center crop will be slightly random
    ctvol, randpad6val = rand_pad(ctvol)
    
    #Crop out the pieces: right lung, heart, and left lung
    ctvol_dict = triple_crop_3d_fixed(ctvol, max_slices, max_side_length, from_seg,
                                      volume_acc, extrema)
    
    #Do data augmentation on each crop separately
    ctvol_dict_aug = {}
    for key in ctvol_dict:
        piece = ctvol_dict[key]
        #Flip randomly (can't rotate because now it's rectangular e.g. [1, 134, 3, 308, 154]
        piece, flip_vector = rand_flip(piece)
        #Make contiguous array to avoid Pytorch error
        piece = np.ascontiguousarray(piece)
        ctvol_dict_aug[key] = piece
    return ctvol_dict_aug

def rand_pad(ctvol):
    """Introduce random padding between 0 and 15 pixels on each of the 6 sides
    of the <ctvol>"""
    randpad6val = [0,0,0,0,0,0]
    #Uncomment the following lines in order to do random padding:
    # randpad6val = np.random.randint(low=0,high=15,size=(6))
    # ctvol = np.pad(ctvol, pad_width = ((randpad6val[0],randpad6val[1]), (randpad6val[2],randpad6val[3]), (randpad6val[4], randpad6val[5])),
    #                      mode = 'constant', constant_values = np.amin(ctvol))
    return ctvol, randpad6val
    
def rand_flip(ctvol):
    """Flip <ctvol> along a random axis with 50% probability"""
    flip_vector = [0,0,0] #records which axis was flipped, if any
    if np.random.randint(low=0,high=100) < 50:
        chosen_axis = np.random.randint(low=0,high=3) #0, 1, and 2 are axis options
        ctvol =  np.flip(ctvol, axis=chosen_axis)
        flip_vector[chosen_axis] = 1
    return ctvol, flip_vector

def rand_rotate(ctvol):
    """Rotate <ctvol> some random amount axially with 50% probability
    
    New 2/2/2021: if the ctvol is a cube, rotate <ctvol> some random amount
    around each of its axes with 50% probability"""
    #Notes related to the 2/2/2021 update:
    #rot_vector records amount of axial rotation, if any.
    #note that if you decide to do semi-supervised learning again, you should
    #update rot_vector so that it also indicates rotation around other axes.
    #Also if you want to do anything with a mask loss with square ctvols, you
    #need to update rot_vector to record rotation around other axes (since
    #attention gr truth needs to be equivalently transformed).
    
    #original function:
    rot_vector = [0,0,0] 
    if np.random.randint(low=0,high=100) < 50:
        chosen_k = np.random.randint(low=1,high=4) #1, 2, or 3 rotations
        ctvol = np.rot90(ctvol, k=chosen_k, axes=(1,2)) #axes ONE/TWO
        rot_vector[chosen_k-1] = 1 #indices 0, 1, or 2
    
    #new 2/2/2021: for square ctvols perform additional rotations
    #WARNING: these additional rotations are NOT currently recorded in rot_vector!!!
    if (ctvol.shape[0]==ctvol.shape[1]==ctvol.shape[2]):
        if np.random.randint(low=0,high=100) < 50:
            ctvol = np.rot90(ctvol, k=np.random.randint(low=1,high=4), axes=(0,1)) #axes ZERO/ONE
        if np.random.randint(low=0,high=100) < 50:
            ctvol = np.rot90(ctvol, k=np.random.randint(low=1,high=4), axes=(0,2)) #axes ZERO/TWO
    
    return ctvol, rot_vector

#####################
# Extrema Functions #-----------------------------------------------------------
#####################
def load_extrema(extrema_path):
    """Load the extrema file, fix it, and return"""
    print('\tLoading the extrema file')
    extrema = pd.read_csv(extrema_path,header=0,index_col=0)
    return fix_extrema(extrema)

def get_clean_extrema_for_ctvol(ctvol, max_slices, max_side_length, volume_acc, extrema):
    """Return ctvol extrema"""
    #Extract the extrema (the lung bbox coordinates for cropping)
    sup_axis0min = extrema.at[volume_acc,'sup_axis0min']
    inf_axis0max = extrema.at[volume_acc,'inf_axis0max']
    ant_axis1min = extrema.at[volume_acc,'ant_axis1min']
    pos_axis1max = extrema.at[volume_acc,'pos_axis1max']
    rig_axis2min = fix_right_border(extrema.at[volume_acc,'rig_axis2min'], extrema)
    lef_axis2max = fix_left_border(extrema.at[volume_acc,'lef_axis2max'], extrema)
    
    #If the size will be too small/big, expand/contract the coordinates to
    #capture more/less of the available volume
    sup_axis0min, inf_axis0max = adjust_coordinates(sup_axis0min, inf_axis0max, max_slices, ctvol.shape[0])
    ant_axis1min, pos_axis1max = adjust_coordinates(ant_axis1min, pos_axis1max, max_side_length, ctvol.shape[1])
    rig_axis2min, lef_axis2max = adjust_coordinates(rig_axis2min, lef_axis2max, max_side_length, ctvol.shape[2])
    
    return sup_axis0min, inf_axis0max, ant_axis1min, pos_axis1max, rig_axis2min, lef_axis2max

def fix_right_border(rig_axis2min, extrema): #Done with testing
    """In a few cases, the right lung is missing and the right border is thus
    too large (greater than ~130). In these cases, replace the coordinate of
    the right border with the right border training set mean (~55)"""
    if rig_axis2min > 130:
        train = extrema[extrema['Subset_Assigned']=='imgtrain']
        assert (train.shape[0]==25355 or train.shape[0]==2) #for testing
        return int(round(np.mean(train.loc[:,'rig_axis2min'])))
    return rig_axis2min

def fix_left_border(lef_axis2max, extrema): #Done with testing
    """In a few cases, the left lung is missing and the left border is thus
    too small (less than ~300). In these cases, replace the coordinate of the
    left border with the left border training set mean (~390)"""
    if lef_axis2max < 300:
        train = extrema[extrema['Subset_Assigned']=='imgtrain']
        assert (train.shape[0]==25355 or train.shape[0]==2) #for testing
        return int(round(np.mean(train.loc[:,'lef_axis2max'])))
    return lef_axis2max

def fix_extrema(extrema): #Done with testing
    """Replace np.nan, np.inf, and -np.inf values with integers based on axis.
    Also ensure all relevant columns are integers."""
    print('\tFixing the extrema file')
    def is_forbidden(value):
        """Check if a value is np.nan, np.inf, or -np.inf. (Cannot do 'in'
        a list that includes np.nan because np.nan!=np.nan)"""
        if np.isnan(value):
            return True
        if value in [np.inf,-np.inf]:
            return True
        return False
    for volume_acc in extrema.index.values.tolist():
        if is_forbidden(extrema.at[volume_acc,'sup_axis0min']):
            extrema.at[volume_acc,'sup_axis0min'] = 0
        if is_forbidden(extrema.at[volume_acc,'inf_axis0max']):
            extrema.at[volume_acc,'inf_axis0max'] = extrema.at[volume_acc,'shape0']
        if is_forbidden(extrema.at[volume_acc,'ant_axis1min']):
            extrema.at[volume_acc,'ant_axis1min'] = 0
        if is_forbidden(extrema.at[volume_acc,'pos_axis1max']):
            extrema.at[volume_acc,'pos_axis1max'] = extrema.at[volume_acc,'shape1']
        if is_forbidden(extrema.at[volume_acc,'rig_axis2min']):
            extrema.at[volume_acc,'rig_axis2min'] = 0
        if is_forbidden(extrema.at[volume_acc,'lef_axis2max']):
            extrema.at[volume_acc,'lef_axis2max'] = extrema.at[volume_acc,'shape2']
    for colname in ['sup_axis0min','inf_axis0max','ant_axis1min','pos_axis1max',
                   'rig_axis2min','lef_axis2max','shape0','shape1','shape2']:
        extrema = extrema.astype({colname: 'int32'})
    return extrema

def adjust_coordinates(minline, maxline, goal, axis_length): #Done with testing
    """Given minimum coordinate <minline> and maximum coordinate <maxline>:
    if the distance between <minline> and <maxline> is less than <goal>,
    expand the distance between them by decreasing <minline> and increasing
    <maxline> such that <maxline> - <minline> = <goal>.
    if the distance between <minline> and <maxline> is bigger than <goal>,
    contract the distance between them by increasing <minline> and decreasing
    <maxline> such that <maxline> - <minline> = <goal>.
    Also, ensure that <minline> is not less than 0 (outside the volume) and
    that <maxline> is not greater than <axis_length> (outside the volume)"""
    verbose = False
    if verbose: print('minline input',minline,'\nmaxline input',maxline)
    if verbose: print('goal distance',goal,'\naxis length',axis_length)
    length_covered = maxline - minline
    if verbose: print('length_covered',length_covered)
    length_needed = goal - length_covered
    if verbose: print('length_needed',length_needed)
    add_to_max = int(round(length_needed/2.0))
    if verbose: print('add_to_max',add_to_max)
    sub_fr_min = length_needed - add_to_max
    if verbose: print('sub_fr_min',sub_fr_min)
    
    #Calculate new minline and maxline
    if verbose: print('first minline_new:',minline,'-',sub_fr_min,'=',minline - sub_fr_min)
    if verbose: print('first maxline_new:',maxline,'+',add_to_max,'=',maxline + add_to_max)
    minline_new = minline - sub_fr_min
    maxline_new = maxline + add_to_max
    
    #Check that maxline_new doesn't exceed the size of the volume
    if maxline_new > axis_length:
        initial_maxline_new = maxline_new #because we are going to overwrite maxline_new
        if verbose: print('maxline exceeds axis length. new maxline:',axis_length)
        if verbose: print('\tnew minline:',minline_new,'-',abs(initial_maxline_new - axis_length),'=',minline_new - abs(initial_maxline_new - axis_length))
        maxline_new = axis_length
        minline_new = minline_new - abs(initial_maxline_new - axis_length)
    
    #Check that minline_new doesn't go below 0
    if minline_new < 0:
        initial_minline_new = minline_new #because we are going to overwrite minline_new
        if verbose: print('minline is below 0. new minline:',0)
        if verbose: print('\tnew maxline is the min of:',axis_length,',',maxline_new + abs(initial_minline_new))
        minline_new = 0
        maxline_new = min(axis_length, maxline_new + abs(initial_minline_new))
    return minline_new, maxline_new

###########################################
# 2019_10 Dataset Preprocessing Sequences #-------------------------------------
###########################################
def prepare_ctvol_2019_10_dataset(ctvol, volume_acc, extrema, data_augment, 
                pixel_bounds, max_slices, max_side_length, num_channels,
                crop_type, selfsupervised, from_seg):
    """Pad, crop, possibly augment, reshape to correct
    number of channels,
    cast to torch tensor (to speed up subsequent operations),
    Clip Hounsfield units, normalize pixel values, center on the
    ImageNet mean, 
    and return as a torch tensor (for crop_type='single') or a dict of torch
    tensors (for crop_type='triple')
    
    Variables:
    <pixel_bounds> is a list of ints e.g. [-1000,200] Hounsfield units. Used for
        pixel value clipping and normalization.
    <max_slices>: int. number of axial slices to keep. e.g. 402 or 420
    <max_slide_length>: int. number of sagittal and coronal slices to
        keep. e.g. 420
    <num_channels> is an int indicating the number of channels to reshape the
        image to. e.g. 3 to reshape the grayscale volume into a volume of
        3-channel images. 
        == 3 if the model uses a feature extractor pretrained on ImageNet.
            Then the returned volume will have 4 dimensions
        == 1 if the model uses only 3D convolutions.
            Then the returned volume with have 3 dimensions
    <crop_type>: if 'single' then return the volume.
        if 'triple' then return a dict of three volumes corresponding to
            the right lung, heart, and left lung.
    <selfsupervised>: if True then return a list of 1s and 0s indicating the
        kind of data augmentation that was performed. Only implemented for
        crop_type = 'single' (and the augmentation vector only has nonzero
        entries if data_augment is True)
    
    Variables needed if you wish to make use of the lung bbox information
    from the lung segmentation:
    <from_seg>: if True, then perform the cropping based on lung bounding box
        coordinates available in <extrema>
    <volume_acc>: scan identifier. this is needed when <from_seg> is True.
        e.g. 'RHAA12345_6.npz' which allows us to load the lung bbox coordinates
        for the correct CT scan
    <extrema>: pandas dataframe. This df includes the lung bbox coordinates
        for all the scans. Index is: full_filename_npz (e.g. 'RHAA12345_6.npz')
        Columns are: Accession,MRN,Set_Assigned,Set_Should_Be,Subset_Assigned,
        sup_axis0min,inf_axis0max,ant_axis1min,pos_axis1max,rig_axis2min,
        lef_axis2max,shape0,shape1,shape2"""
    assert num_channels == 3 or num_channels == 1
    
    #Padding to minimum size [max_slices, max_side_length, max_side_length]
    if not from_seg: #if you are not using the lung bbox then you will FIRST pad
        ctvol = pad_volume(ctvol, max_slices, max_side_length)
    
    #Cropping, and data augmentation if indicated
    if crop_type == 'single':
        if data_augment is True:
            ctvol, flip_vector, rot_vector, randpad6val = single_crop_3d_augment(ctvol, max_slices, max_side_length, from_seg, volume_acc, extrema)
        else:
            flip_vector = [0,0,0]; rot_vector = [0,0,0]; randpad6val=[0,0,0,0,0,0] #no data aug was performed
            if not from_seg:
                ctvol = single_crop_3d_fixed(ctvol, max_slices, max_side_length)
            else: #seg/lung bbox
                ctvol = single_crop_3d_fixed_from_seg(ctvol, max_slices, max_side_length, volume_acc, extrema)
        #Reshape to 3 channels if indicated
        if num_channels == 3:
            ctvol = reshape_3_channels(ctvol)
        #Cast to torch tensor and deal with pixel values
        output = torchify_pixelnorm_pixelcenter(ctvol, pixel_bounds)
    
    #TODO: include flip_vector in the triple crop setup!
    elif crop_type == 'triple': #right lung, heart, left lung
        assert not selfsupervised #NOTE: flip_vector is NOT currently included in the triple crop setup!
        flip_vector = [None,None,None]; rot_vector=[None,None,None]; randpad6val=[None,None,None,None,None,None] #selfsupervised not implemented!
        assert not mask #NOTE: mask is NOT currently included in triple crop setup!
        if data_augment is True:
            ctvol_dict = triple_crop_3d_augment(ctvol, max_slices, max_side_length, from_seg, volume_acc, extrema)
        else:
            ctvol_dict = triple_crop_3d_fixed(ctvol, max_slices, max_side_length, from_seg, volume_acc, extrema)
        #Reshape to 3 channels if indicated:
        if num_channels == 3:
            output = {}
            for key in ctvol_dict.keys():
                output[key] = reshape_3_channels(ctvol_dict[key])
        else: #num_channels == 1
            output = ctvol_dict
        #Cast to torch tensor and deal with pixel values
        for key in output.keys():
            output[key] = torchify_pixelnorm_pixelcenter(output[key], pixel_bounds)
    
    auglabel = torch.Tensor(flip_vector+rot_vector) #e.g. [0,0,1,0,1,0]
    return output, auglabel, randpad6val
