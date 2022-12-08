#run10_AxialNetMask_orangebluelinefigs_test_wholedata.py
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

import timeit

from models import custom_models_mask
from load_dataset import custom_datasets

from plot import viz_orangebluelines

#run10: Make the orangeblueline figures using the test set and the final full
#data AxialNet mask loss model. Some of these orangeblueline figures are shown
#in Figure S4.

if __name__=='__main__':
    tot0 = timeit.default_timer()
    m = viz_orangebluelines.MakeLineVisualizations(descriptor='2020-12-12_AxialNetMask-TestSet',
                custom_net = custom_models_mask.AxialNet_Mask,
                custom_net_args = {'n_outputs':80,'slices':135},
                batch_size = 1, device = 0,
                
                #TODO: replace this with the correct path to the final parameters
                full_params_dir ='/home/rlb61/data/img-hiermodel2/results/2020-12-12_AxialNetMask_WholeData_dilateFalse_nearest/params/AxialNetMask_WholeData_dilateFalse_nearest_epoch4',
                
                dataset_class = custom_datasets.CTDataset_2019_10,
                dataset_args = {
                    'verbose':False,
                    'loss_string':'bce',
                    'label_type_ld':'location_disease_0323',
                    'genericize_lung_labels':True,
                    'label_counts':{'mincount_heart':200,
                                'mincount_lung':125},
                    'view':'axial',
                    'use_projections9':False,
                    'volume_prep_args':{
                                'pixel_bounds':[-1000,800],
                                'num_channels':3,
                                'crop_type':'single',
                                'selfsupervised':False,
                                'from_seg':False},
                    'attn_gr_truth_prep_args':{
                                'dilate':'',
                                'downsamp_mode':''},
                    'selected_note_acc_files':{'train':'','valid':'','test':''}},
                
                #TODO: replace this with the path to the desired results directory
                #in which to store the orangebluelinefigs
                results_dir='/home/rlb61/data/img-hiermodel2/results/2020-12-12_OrangeBlueLineFigs-for-2020-12-12_AxialNetMask-TestSet')
    m.run_model()
    tot1 = timeit.default_timer()
    print('Total Time', round((tot1 - tot0)/60.0,2),'minutes')
