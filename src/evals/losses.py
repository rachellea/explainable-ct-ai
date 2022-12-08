#losses.py
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

import math
import numpy as np
import torch, torch.nn as nn

def calculate_loss(loss_string, out, gr_truth,
                   train_labels_df, device, epoch, training, batch, loss_args):
    """Return the loss.
    
    Variables:
    <loss_string>: a string defining the loss function
    <out>: either a Tensor (the direct final output of the model) or a dict
        of Tensors
    <gr_truth>: a Tensor for the classification ground truth
    <train_labels_df>: needed for 'bce-weighted'. Training labels dataframe
    <device>: the device
    <epoch>: int, the epoch number. Needed for losses with a schedule.
    <training>: True when the model is training, needed for 'bce-selfsup-schedule'
    <batch>: the entire batch dictionary
    <loss_args>: dict of additional arguments to be passed to the final loss
        function. This is used in the mask loss to determine the lambda_val
        i.e. the relative weighting of the classification part and the
        mask-based part"""
    if 'bce' == loss_string:
        loss_func = nn.BCEWithLogitsLoss() #includes application of sigmoid for numerical stability
        return loss_func(out, gr_truth)
    elif 'bce-text' == loss_string:
        loss_func = nn.BCEWithLogitsLoss() #includes application of sigmoid for numerical stability
        return loss_func(out, batch['textlabel'].to(device))
    elif 'bce-weighted' == loss_string:
        loss_func = nn.BCEWithLogitsLoss(pos_weight=calculate_pos_weight(train_labels_df, device))
        return loss_func(out, gr_truth)
    elif 'mse' == loss_string: #Mean Square Error Loss
        loss_func = nn.MSELoss()
        return loss_func(out, gr_truth)
    elif 'bce-selfsup-schedule' in loss_string:
        #no reduction, so that reduction can be applied after the
        #fliprot labels have had a lambda applied to them
        loss_func = nn.BCEWithLogitsLoss(reduction='none')
        loss_nonreduced = loss_func(out, gr_truth)
        return reduce_bce_selfsup_schedule(loss_nonreduced, epoch, training, device, loss_string)
    elif 'BodyLocationAttn3Mask-loss' == loss_string:
        return calculate_BodyLocationAttn3Mask_loss(out, gr_truth, batch, device)
    elif 'BodyDiseaseSpatialAttn4Mask-loss' == loss_string:
        return calculate_BodyDiseaseSpatialAttn4Mask_loss(out, gr_truth,
            batch, device)
    elif 'AxialNet_Mask-loss' ==  loss_string:
        return calculate_AxialNet_Mask_loss(out, gr_truth,
            batch, device, **loss_args)
    elif 'AxialNet_Mask-loss-L2Variant' == loss_string:
        return calculate_AxialNet_Mask_loss_L2Variant(out, gr_truth,
            batch, device)
    else:
        assert False, loss_string+' not available'

def calculate_pred_np_and_gr_truth_np(loss_string, out, gr_truth, batch):
    """Return pred_np (the model's predictions as a np array) and gr_truth_np
    (the ground truth as a np array)
    <gr_truth> is the abnormalities ground truth
    <batch> is only needed for 'bce-text'
    """
    sigmoid = torch.nn.Sigmoid()
    if loss_string == 'mse':
        #first binarize 'out' and 'gr_truth' so that anything below 1 is
        #zero and anything >=1 is 1
        out_bin = out.data.masked_fill_(out.data<1, 0).masked_fill_(out.data >=1, 1)
        gr_truth_bin = gr_truth.masked_fill_(gr_truth<1, 0).masked_fill_(gr_truth>=1, 1)
        pred_np = (out_bin).detach().cpu().numpy()
        gr_truth_np = gr_truth_bin.detach().cpu().numpy()
    elif (('bce-selfsup-schedule' in loss_string) or (loss_string in ['bce','bce-l1','bce-weighted'])):
        pred_np = sigmoid(out.data).detach().cpu().numpy()
        gr_truth_np = gr_truth.detach().cpu().numpy()
    elif loss_string == 'bce-text':
        pred_np = sigmoid(out.data).detach().cpu().numpy()
        gr_truth_np = batch['textlabel'].detach().cpu().numpy()
    elif loss_string in ['BodyLocationAttn3Mask-loss','BodyDiseaseSpatialAttn4Mask-loss',
                         'AxialNet_Mask-loss','AxialNet_Mask-loss-L2Variant']:
        #the classification loss is just 'bce' but we need a special
        #elif because 'out' is a dictionary rather than just the raw
        #predictions.
        pred_np = sigmoid(out['out'].data).detach().cpu().numpy()
        gr_truth_np = gr_truth.detach().cpu().numpy()
    elif loss_string == 'bce-fancy':
        pred_np = (out.data).detach().cpu().numpy() #sigmoid has already been applied
        gr_truth_np = gr_truth.detach().cpu().numpy()
    return pred_np, gr_truth_np

#########################
# For BCE Weighted Loss #-------------------------------------------------------
#########################
def calculate_pos_weight(train_labels_df, device):
    """When the loss is 'bce-weighted', initialize self.pos_weights which
    contains the weighting that should be applied to positives in each
    class in the BCEWithLogitsLoss. The weighting is based on the ratio
    of negatives to positives in the training set."""
    pos_count = train_labels_df.sum(axis = 0) #count of positives per class
    grandtotal = train_labels_df.shape[0] #number of scans
    neg_count = grandtotal - pos_count
    #https://pytorch.org/docs/master/generated/torch.nn.BCEWithLogitsLoss.html
    #If a dataset contains 100 positive and 300 negative examples of a single
    #class, then pos_weight for the class should be equal to 300/100 = 3
    pos_weight_series = neg_count/pos_count
    #Now cap the positive weights to range between 1 and 10,
    #(if I just leave the positive weights the way they are, the training
    #goes badly and the model doesn't learn)
    pos_weight_series[pos_weight_series>10] = 10.0
    pos_weight_series[pos_weight_series<1] = 1.0
    print('pos_weights_df is\n',pos_weight_series)
    pos_weight = torch.Tensor(np.squeeze(pos_weight_series.values)).to(device)
    return pos_weight

###############
# Mask Losses #-----------------------------------------------------------------
###############
def calculate_BodyDiseaseSpatialAttn4Mask_loss(out, gr_truth, batch, device):
    """See documentation for calculate_classification_and_dzattn_loss()
    Here, the attn is the output of a soft trainable attention mechanism."""
    #The attention has shape [slices, n_outputs, 1, 6, 6] where n_outputs = n_outputs_heart + n_outputs_lung
    #Convert to shape [n_outputs, slices, 6, 6]:
    attn = out['attn'].squeeze().transpose(0,1)
    return calculate_classification_and_dzattn_loss(out, gr_truth, attn, batch, device)

def calculate_AxialNet_Mask_loss(out, gr_truth, batch, device, lambda_val):
    """See documentation for calculate_classification_and_dzattn_loss()
    Here, the attn is (roughly speaking) from Grad-CAM.
    More specifically, the attn comes from the intermediate
    calculation in the final FC layer. See AxialNet_Mask in custom_models_mask.py
    for details about how the attn is calculated."""
    disease_reps = out['disease_reps'] #shape [n_outputs, slices, 16, 6, 6]
    #Squish together the feature dimension by summing
    attn = torch.sum(disease_reps,dim=2) #out shape [n_outputs, slices, 6, 6]
    return calculate_classification_and_dzattn_loss(out, gr_truth, attn, batch, device, lambda_val)

def calculate_AxialNet_Mask_loss_L2Variant(out, gr_truth, batch, device):
    """See documentation for calculate_AxialNet_Mask_loss()"""
    disease_reps = out['disease_reps']
    attn = torch.sum(disease_reps,dim=2)
    return calculate_classification_and_dzattn_loss_L2Variant(out, gr_truth, attn, batch, device)

def calculate_classification_and_dzattn_loss(out, gr_truth, attn, batch, device, lambda_val):
    """This loss has a classification part and an attention part. The
    attention part can come from soft trainable attention or from post-hoc
    attention like Grad-CAM.
    
    The classification part assumes we've only included heart labels and
    generic lung labels (i.e. NOT right lung vs left lung, just 'lung').
    However we still make use of right vs. left knowledge for a lung disease
    in calculating the attention part of the loss.
    In the attention part of the loss the location information is used to
    determine what locations the disease-specific attention is allowed to
    look at. e.g. if there is atelectasis only in the left lung then the
    attention for atelectasis for that scan should be only in the place
    demarcated as left lung in the segmentation ground truth.
    Furthermore, if there is NO atelectasis present, then the attention
    for atelectasis should all be zero."""
    #First calculate the classification loss
    bce = nn.BCEWithLogitsLoss()
    classification_loss = bce(out['out'], gr_truth) #e.g. gr_truth has shape [1,81]
    
    #Next calculate the attention loss.
    attn_gr_truth = flip_and_rotate_attn_gr_truth(batch, device)
    assert attn.shape == attn_gr_truth.shape #sanity check
    #Penalize model for any 'pro-disease' (high) attention values outside of the
    #allowed regions (i.e. the model cannot increase the disease score
    #using outside regions)
    outside_allowed_regions_attn = attn[attn_gr_truth==0].flatten()
    attention_loss_outside = bce(outside_allowed_regions_attn.unsqueeze(0), torch.zeros(1,outside_allowed_regions_attn.shape[0]).to(device))
    
    #Overall loss
    total_loss = classification_loss+(lambda_val*attention_loss_outside)
    return total_loss

def flip_and_rotate_attn_gr_truth(batch, device): #TODO TEST THIS
    """Return <attn_gr_truth> flipped and/or rotated according to the
    transformations specified in <batch>.
    
    In unit_tests.py see check_if_torch_and_np_have_equivalent_flip_and_rotate()
    which verifies that torch.flip and np.flip have the same behavior, and
    torch.rot90 and np.rot90 have the same behavior. This is important because
    the CT scan flipping and rotating uses numpy functions, but here we need
    to use torch functions because we are working with a Tensor on the GPU."""
    #The attn_gr_truth is in the default axial orientation. We need to make it
    #match the orientation of the CT volume, which may have been flipped or
    #rotated in the preprocessing.
    attn_gr_truth = batch['attn_gr_truth'].squeeze().to(device) #out shape [n_outputs, slices, 6, 6] 

    #Note that the CT volume had shape [slices, 6, 6] when it was being
    #flipped and rotated. The attn_gr_truth has shape  [n_outputs, slices, 6, 6].
    #We need to apply the transformations to the [slices, 6, 6] part.
    
    #First figure out what augmentation was applied to the scan
    #Need to select [0] from batch['auglabel'] because we are assuming a
    #batch size of one.
    flip_vector = np.array([int(x) for x in batch['auglabel'][0][0:3].tolist()])
    rot_vector = np.array([int(x) for x in batch['auglabel'][0][3:].tolist()])
    
    #Flip
    if 1 in flip_vector:
        chosen_axis = int(np.where(flip_vector==1)[0][0])
        for n_output in range(attn_gr_truth.shape[0]):
            attn_gr_truth[n_output,:,:,:] = torch.flip(attn_gr_truth[n_output,:,:,:], dims=(chosen_axis,))
    
    #Rotate
    if 1 in rot_vector:
        chosen_k = int(np.where(rot_vector==1)[0][0]+1) #indices are 0, 1, 2 for k equal to 1, 2, or 3 respectively
        for n_output in range(attn_gr_truth.shape[0]):
            attn_gr_truth[n_output,:,:,:] = torch.rot90(attn_gr_truth[n_output,:,:,:], k=chosen_k, dims=(1,2))
    
    return attn_gr_truth

def calculate_classification_and_dzattn_loss_L2Variant(out, gr_truth, attn, batch, device):
    """See documentation and comments in
    calculate_classification_and_dzattn_loss()
    
    Difference: this uses L2 loss (MSELoss) to encourage the forbidden regions
    to be as close to zero as possible.
    (The other function uses cross entropy, which just says the forbidden
    regions cannot contribute positively to the disease classification.)"""
    #First calculate the classification loss
    bce = nn.BCEWithLogitsLoss()
    classification_loss = bce(out['out'], gr_truth) #e.g. gr_truth has shape [1,81]
    
    #Next calculate the attention loss.
    attn_gr_truth = flip_and_rotate_attn_gr_truth(batch, device)
    assert attn.shape == attn_gr_truth.shape #sanity check
    
    #Penalize model for any high attention values outside of the
    #allowed regions. L2 Variant, implemented with MSE Loss.
    outside_allowed_regions_attn = attn[attn_gr_truth==0].flatten()
    attention_loss_outside = torch.norm(outside_allowed_regions_attn, p=2) #this is in the ballpark of 1.14
    
    #Overall loss
    lambda_val = 1.0/3.0
    total_loss = classification_loss+(lambda_val*attention_loss_outside)
    return total_loss

###############################################
# Schedules for self-supervised learning loss #---------------------------------
###############################################
def reduce_bce_selfsup_schedule(loss_nonreduced, epoch, training, device, loss_string):
    """If the loss is 'bce-selfsup-schedule' then that means we want to do
    selfsupervised learning with a schedule on a lambda value that
    multiplies the fliprot labels, so that eventually the fliprot labels
    don't contribute to the loss at all.
    Notice that for the validation set the fliprot labels are never included.
    <loss_nonreduced> is the output of BCEWithLogitsLoss with
        reduction=='none'. loss_nonreduced.shape is for example [1, 89]
    <epoch> is the epoch number. epoch is an int e.g. 0
    <training> is True when the model is training"""
    if training:
        lambda_val = get_lambda_val(loss_string,epoch)
        #Apply weights based on lambda_val, to downweight the fliprot labels
        #which are the last 6 labels in the vector
        shape = list(loss_nonreduced.shape)
        weights = np.ones(loss_nonreduced.shape,dtype='float')
        weights[:,-6:] = lambda_val
        loss_nonreduced = torch.Tensor(weights).to(device)*loss_nonreduced
    
    if not training:
        #If it's the validation set, then don't include the fliprot labels
        #at all, ever, because we want early stopping to be based on the disease
        #labels only:
        loss_nonreduced = loss_nonreduced[:,0:-6]
        assert list(loss_nonreduced.shape)[1] in [83,132]
    
    #Reduce
    loss_reduced = loss_nonreduced.mean(dim=1)
    return loss_reduced.squeeze()

def get_lambda_val(loss_string,epoch):
    if loss_string == 'bce-selfsup-schedule100':
        lambda_val = _selfsup_schedule100(epoch)
    elif loss_string == 'bce-selfsup-schedule101':
        lambda_val = _selfsup_schedule101(epoch)
    elif loss_string == 'bce-selfsup-schedule102':
        lambda_val = _selfsup_schedule102(epoch)
    
    #End epoch 20
    elif loss_string == 'bce-selfsup-schedule201':
        lambda_val = _calculate_log_selfsup_schedule(startlambda=1, endepoch=20, epoch=epoch)
    elif loss_string == 'bce-selfsup-schedule202':
        lambda_val = _calculate_log_selfsup_schedule(startlambda=2, endepoch=20, epoch=epoch)
    elif loss_string == 'bce-selfsup-schedule204':
        lambda_val = _calculate_log_selfsup_schedule(startlambda=4, endepoch=20, epoch=epoch)
    
    #End epoch 30
    elif loss_string == 'bce-selfsup-schedule301':
        lambda_val = _calculate_log_selfsup_schedule(startlambda=1, endepoch=30, epoch=epoch) 
    elif loss_string == 'bce-selfsup-schedule302':
        lambda_val = _calculate_log_selfsup_schedule(startlambda=2, endepoch=30, epoch=epoch)
    elif loss_string == 'bce-selfsup-schedule304':
        lambda_val = _calculate_log_selfsup_schedule(startlambda=4, endepoch=30, epoch=epoch)
    
    #End epoch 40
    elif loss_string == 'bce-selfsup-schedule401':
        lambda_val = _calculate_log_selfsup_schedule(startlambda=1, endepoch=40, epoch=epoch)
    elif loss_string == 'bce-selfsup-schedule402':
        lambda_val = _calculate_log_selfsup_schedule(startlambda=2, endepoch=40, epoch=epoch)
    elif loss_string == 'bce-selfsup-schedule404':
        lambda_val = _calculate_log_selfsup_schedule(startlambda=4, endepoch=40, epoch=epoch)
    return lambda_val

def _selfsup_schedule100(epoch):
    """Steps schedule. Goes from 1.0 before epoch 20, to 0.2 for epochs 20 to
    30, to 0 after epoch 30"""
    if epoch <= 20:
        lambda_val = 1.0
    elif (20 < epoch < 30):
        lambda_val = 0.2
    elif (30 <= epoch):
        lambda_val = 0
    return lambda_val

def _selfsup_schedule101(epoch):
    """Linear schedule. Decreases from 1 at epoch 0, down to 0 at epoch 30.
    Then stays at 0."""
    lambda_val = ((-1.0/30.0)*epoch)+1
    return max(lambda_val, 0)

def _selfsup_schedule102(epoch):
    """Linear schedule. Decreases from 2 at epoch 0, down to 0 at epoch 30.
    Then stays at 0."""
    lambda_val = ((-1.0/15.0)*epoch)+2
    return max(lambda_val, 0)

def _calculate_log_selfsup_schedule(startlambda, endepoch, epoch):
    """Return the lambda value at <epoch> for a self-supervised schedule
    based on <startlambda> which is the lambda value for epoch 0, and
    <endepoch> which is the epoch at which the lambda value should become 0.
    The point (0, startlambda) is the desired approximate y-intercept
    and the point (endepoch, 0) is the desired x-intercept.
    Note that since ln(0) is not defined, epoch 0 is defined explicitely to
    take value <startlambda>.
    The equation is of the form y = m ln(x) + b
    or, lambda_val = m ln(epoch) + b"""
    if epoch == 0:
        return startlambda
    else:
        b = startlambda
        m = (-1*startlambda)/math.log(endepoch)
        lambda_val = m*math.log(epoch) + b
        return max(lambda_val, 0)
    