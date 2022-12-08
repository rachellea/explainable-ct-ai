#2021-01-29-visualizing_custom_datasets_text.py
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
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.ioff()

from src.load_dataset import custom_datasets_text

def run():
    dataset_args = {'setname':'valid',
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
                                    'num_channels':1,
                                    'crop_type':'single',
                                    'selfsupervised':False,
                                    'from_seg':False},
                        'attn_gr_truth_prep_args':{
                                'dilate':None,
                                'downsamp_mode':None},
                        #Paths
                        'selected_note_acc_files':{'train':'/home/rlb61/data/img-hiermodel2/data/RADChestCT_DEID/predefined_subsets/2020-01-10-imgtrain_random2000_DEID.csv',
                                                   'valid':'/home/rlb61/data/img-hiermodel2/data/RADChestCT_DEID/predefined_subsets/2020-01-10-imgvalid_a_random1000_DEID.csv'},
                        'ct_scan_path':'/scratch/rlb61/2019-10-BigData-DEID',
                        'ct_scan_projections_path':'/scratch/rlb61/2020-04-15-Projections-DEID',
                        'key_csvs_path':'/home/rlb61/data/img-hiermodel2/data/RADChestCT_DEID/',
                        'segmap_path':'/storage/rlb61-ct_images/vna/rlb61/2019-10-BigData-segmaps-DEID/'}
    dataset = custom_datasets_text.CTDataset_2019_10_with_Text(**dataset_args)
    meep = dataset[0]
    torch.save(meep['data'],'data.pt')
    
    for idx in range(200,250):
        if meep['textlabel'][idx] == 1:
            plt.figure(figsize=(8, 8))
            plt.imshow(meep['data'].numpy()[idx,:,:], cmap = plt.cm.gray)
            plt.tight_layout(pad=0)
            plt.gca().set_axis_off()
            plt.savefig('tempfig'+str(idx)+'.png', bbox_inches='tight',pad_inches=0)
            plt.close()

if __name__=='__main__':
    run()