#viz_orangebluelines.py
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
import numpy as np
import pandas as pd
import torch, torch.nn as nn

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.ioff()
params = {'figure.figsize':(6,6),
         'figure.titlesize':20,
         'legend.fontsize':12,
         'legend.title_fontsize':12,
         'axes.labelsize':20,
         'axes.titlesize':20,
         'xtick.labelsize':12,
         'ytick.labelsize':12}
plt.rcParams.update(params)
import seaborn as sns

#Set seeds
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

from plot import main_viz
from attn_analysis import renaming_abnormalities_new

class MakeLineVisualizations(main_viz.MakeVisualizations):
    def viz_dataset(self, dataloader):
        if os.path.exists(os.path.join(self.results_dir,'mega_detailed_summary_df.csv')):
            self.plot_based_on_saved_files()
        else:
            self.plot_based_on_new_calculations(dataloader)
    
    # Methods #-----------------------------------------------------------------
    def plot_based_on_saved_files(self):
        print('Plot based on saved files')
        mega_detailed_summary_df = pd.read_csv(os.path.join(self.results_dir,'mega_detailed_summary_df.csv'),header=0,index_col=0)
        self.plot_fancy_red_and_blue_line_summaries(mega_detailed_summary_df)
    
    def plot_based_on_new_calculations(self, dataloader):
        print('Plot based on new calculations')
        model = self.custom_net(**self.custom_net_args).to(self.device).eval()
        print('For predictions, loading model params from',self.full_params_dir)
        check_point = torch.load(self.full_params_dir)
        model.load_state_dict(check_point['params'])
        epoch = int(self.full_params_dir.split('epoch')[1])
        with torch.no_grad():
            self.iterate_through_batches_and_sum(model, dataloader, epoch)
    
    def iterate_through_batches_and_sum(self, model, dataloader, epoch):
        """Sum across entire dataset"""
        print('Working on iterate_through_batches_and_sum()')
        
        #Initialize mega detailed summary df for making seaborn plots
        num_examples = len(self.dataset_test)
        num_labels = len(self.label_meanings)
        num_slices = self.custom_net_args['slices']
        total_rows = num_examples*num_labels*num_slices
        mega_detailed_summary_df = pd.DataFrame(np.zeros((total_rows,5)),index=[x for x in range(0,total_rows)],columns=['volume_acc','slice_num','label','score','gr_truth'])
        mega_detailed_summary_df['volume_acc']=''
        mega_detailed_summary_df['label']=''
        curr_idx=0
        
        for batch_idx, batch in enumerate(dataloader):
            data = batch['data'].to(self.device) #e.g. shape [1, 135, 3, 420, 420]
            volume_acc = batch['volume_acc'][0] #e.g. 'RHAA12345_6.npz'
            with torch.set_grad_enabled(False):
                out = model(data)
            pred = self.sigmoid(out['out'].data).detach().cpu().numpy()
            gr_truth = batch['gr_truth'].detach().cpu().numpy() # out shape (1, 80)
            slice_pred = np.squeeze(out['x_perslice_scores'].detach().cpu().numpy()) #out shape (80, 135)
            
            #Update the mega_detailed_summary_df
            slice_pred_temp_df = pd.DataFrame(slice_pred,index=self.label_meanings,columns=[x for x in range(1,num_slices+1)])
            gr_truth_temp_df = pd.DataFrame(gr_truth,index=[0],columns=self.label_meanings)
            for label in self.label_meanings:
                for slice_num in range(1,num_slices+1):
                    mega_detailed_summary_df.at[curr_idx,'volume_acc'] = volume_acc
                    mega_detailed_summary_df.at[curr_idx,'slice_num'] = slice_num
                    mega_detailed_summary_df.at[curr_idx,'label'] = label
                    mega_detailed_summary_df.at[curr_idx,'score'] = slice_pred_temp_df.at[label, slice_num]
                    mega_detailed_summary_df.at[curr_idx,'gr_truth'] = gr_truth_temp_df.at[0,label]
                    curr_idx+=1
            del pred, gr_truth, slice_pred, slice_pred_temp_df, gr_truth_temp_df
            torch.cuda.empty_cache()
            if batch_idx %100 == 0: print('Done with',batch_idx)
        
        #Save the mega_detailed_summary_df
        mega_detailed_summary_df.to_csv(os.path.join(self.results_dir,'mega_detailed_summary_df.csv'),header=True,index=True)
        self.plot_fancy_red_and_blue_line_summaries(mega_detailed_summary_df)
    
    def plot_fancy_red_and_blue_line_summaries(self, mega_detailed_summary_df):
        renaming_dict = return_one_renaming_dict()
        fancy_results_dir = os.path.join(self.results_dir,'fancyorangeblue')
        if not os.path.exists(fancy_results_dir):
            os.mkdir(fancy_results_dir)
        #mega_detailed_summary_df columns are ['volume_acc', 'slice_num', 'label', 'score', 'gr_truth']
        mega_detailed_summary_df['Ground Truth'] = 'Absent'
        mega_detailed_summary_df['Ground Truth'][mega_detailed_summary_df['gr_truth']==1.0]='Present'
        mega_detailed_summary_df.rename(columns={'score':'Score','slice_num':'Slice'},inplace=True)
        sigmoid = nn.Sigmoid()
        mega_detailed_summary_df['Probability'] = (sigmoid(torch.Tensor(mega_detailed_summary_df['Score'].values))).data.numpy()
        
        for label_name in self.label_meanings:
            for colname in ['Probability','Score']:
                print('Working on',colname,'for',label_name,'=',renaming_dict[label_name])
                sel_label = mega_detailed_summary_df[mega_detailed_summary_df['label']==label_name]
                num_present = len(set(sel_label[sel_label['Ground Truth']=='Present']['volume_acc']))
                num_absent = len(set(sel_label[sel_label['Ground Truth']=='Absent']['volume_acc']))
                assert (num_present+num_absent)==len(self.dataset_test)
                ax=sns.lineplot(data=sel_label,x='Slice',y=colname,hue='Ground Truth',hue_order=['Absent','Present'])
                plt.title(renaming_dict[label_name]+'\n('+str(num_present)+' Present, '+str(num_absent)+' Absent)')
                plt.xlim([0, 140])
                plt.tight_layout()
                plt.savefig(os.path.join(fancy_results_dir,label_name+'_fancy_redblue_lines_'+colname+'.png'))
                plt.close()
        
def return_one_renaming_dict():
    GV_AND_MEDIA_COLS, GV_AND_MEDIA_COLS_RENAME, HEART_COLS, HEART_COLS_RENAME, LUNG_COLS, LUNG_COLS_RENAME = renaming_abnormalities_new.return_renamers()
    renaming_dict = {}
    for dictionary in [GV_AND_MEDIA_COLS_RENAME,HEART_COLS_RENAME,LUNG_COLS_RENAME]:
        for key in dictionary.keys():
            if key == 'lung_tree_in_bud':
                renaming_dict[key]='Tree-in-Bud' #special case captialization
            else:
                value = dictionary[key]
                #the join/capitalize functionality: 'great vessel calcification' --> 'Great Vessel Calcification'
                renaming_dict[key] = ' '.join([x.capitalize() for x in value.split()])
    return renaming_dict
    