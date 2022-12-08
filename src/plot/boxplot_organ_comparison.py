#boxplot_organ_comparison.py
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

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

"""Make a boxplot from <df> where the columns of <df> are different models,
the rows are different abnormalities, and the values are a performance
metric like AUROC. <rename_dict> is optional; if it is not empty then it
should map from the current column names to new column names and it will
be used to rename the columns before making the figure."""

def clean_df_and_make_boxplot(df, model_filter, organ_filter, rename_dict, plot_title, save_name, xlab, ylab, rotate_xticks, big):
    print(organ_filter)
    clean = prepare_df_completely(df, model_filter, organ_filter, rename_dict, save_name)
    make_boxplot(clean, plot_title, save_name, xlab, ylab, rotate_xticks, big)

def make_boxplot(clean, plot_title, save_name, xlab, ylab, rotate_xticks, big):
    #can also make violinplot
    box_plot = sns.boxplot(data = clean, x = 'Model', y='Performance',
                           hue='SegUsed',  palette='Set3', dodge=False,
                           notch=True)
    plt.title(plot_title)
    plt.xticks(fontsize=7) #I think the default is 10
    
    ##Plot horizontal line at the median of Triple NoSeg model
    #xleft, xright = plt.xlim() #return the current x limits
    #plt.hlines(y = np.median(clean[clean['Model']=='Triple NoSeg']['Performance']), xmin = xleft, xmax = xright, alpha = 0.2)
    
    if xlab!='':
        plt.xlabel(xlab)
    if ylab!='':
        plt.ylabel(ylab)
    if rotate_xticks:
        plt.xticks(rotation=90)
        plt.subplots_adjust(bottom=0.5)
    if big:
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(12,18)
    
    plt.legend(fontsize = 'small')
    plt.savefig(save_name+'.png') #also save png for presentations
    plt.close()

def prepare_df_completely(clean, organ_filter, save_name):
    """Prepare a pandas dataframe <df> to be used as input for making a
    boxplot"""
    organ_filter_idx = []
    for idx in clean.index.values.tolist():
        for string in organ_filter:
            if string in idx:
                organ_filter_idx.append(idx)
    clean = clean.filter(items = organ_filter_idx, axis=0)
    print('clean shape after filtering',clean.shape)
    #Save mean and median
    mean = clean.mean(axis=0)
    median = clean.median(axis=0)
    out = pd.DataFrame(index=mean.index.values.tolist(), columns=['Mean','Median'])
    for model in out.index.values.tolist():
        out.at[model,'Mean'] = mean.at[model]
        out.at[model,'Median'] = median.at[model]
    out.to_csv(save_name+'_MeanMedian.csv')
    #Now finish cleaning the df and return it
    clean = longify_df(clean)
    clean['Performance'] = clean['Performance'].astype('float')
    return clean

def add_independent_col(df):
    """Add columns that summarize all of the individual organ models
    in one column"""
    indep_noseg = df[['LLung NoSeg','Heart NoSeg','RLung NoSeg']]
    indep_noseg = indep_noseg.fillna(value=-1)
    indep_noseg = indep_noseg.max(axis=1)
    
    indep_withseg = df[['LLung WithSeg','Heart WithSeg','RLung WithSeg']]
    indep_withseg = indep_withseg.fillna(value=-1)
    indep_withseg = indep_withseg.max(axis=1)
    
    for locdis in df.index.values.tolist():
        df.at[locdis,'Indep NoSeg'] = indep_noseg.at[locdis]
        df.at[locdis,'Indep WithSeg'] = indep_withseg.at[locdis]
    return df

def longify_df(df):
    """Longify df so that you can use seaborn's grouping functionality"""
    newdf = pd.DataFrame(columns=['Performance','LocDis','SegUsed','Model'],
                         index = [x for x in range(0,df.shape[0]*df.shape[1])])
    newidx = 0
    for locdis in df.index.values.tolist():
        for model in df.columns.values.tolist():
            newdf.at[newidx,'Performance'] = df.at[locdis,model]
            newdf.at[newidx,'LocDis'] = locdis
            if 'NoSeg' in model:
                newdf.at[newidx,'SegUsed'] = 'NoSeg'
            elif 'WithSeg' in model:
                newdf.at[newidx,'SegUsed'] = 'WithSeg'
            newdf.at[newidx,'Model'] = model
            newidx+=1
    return newdf

if __name__ == '__main__':
    perf = {'AUROC':'C:\\Users\\Rachel\\Documents\\CarinLab\\Project_Radiology\\2020-03_March\\2020-03-12-Models-Big-Comparison-Complete\\0312_auroc.csv',
            'Avg Precision':'C:\\Users\\Rachel\\Documents\\CarinLab\\Project_Radiology\\2020-03_March\\2020-03-12-Models-Big-Comparison-Complete\\0312_avg_precision.csv'}
    for perfname in perf.keys():
        print('Working on',perfname)
        dfpath = perf[perfname]
        df = pd.read_csv(dfpath,header=0,index_col=0)
        rename_dict = {
            '2020-02-20_locdis_singlevol_epoch38':'Single NoSeg',
            '2020-02-26_locdis_triplevol_withseg_epoch24':'Triple WithSeg',
            '2020-02-26_locdis_singlevol_withseg_epoch55':'Single WithSeg',
            '2020-03-04_locdis_triplevol_noseg_oneorgan_heart_epoch17':'Heart NoSeg',
            '2020-03-03_locdis_triplevol_noseg_epoch26':'Triple NoSeg',
            '2020-03-04_locdis_triplevol_withseg_oneorgan_heart_epoch45':'Heart WithSeg',
            '2020-03-05_locdis_triplevol_withseg_oneorgan_left_lung_epoch39':'LLung WithSeg',
            '2020-03-05_locdis_triplevol_noseg_oneorgan_left_lung_epoch62':'LLung NoSeg',
            '2020-03-09_locdis_triplevol_withseg_oneorgan_right_lung_epoch55':'RLung WithSeg',
            '2020-03-09_locdis_triplevol_noseg_oneorgan_right_lung_epoch61':'RLung NoSeg'}
        
        #Overall plot
        model_all_filter = ['Single NoSeg','Indep NoSeg','Triple NoSeg','Single WithSeg','Indep WithSeg','Triple WithSeg']
        all_filter = ['']
        clean_df_and_make_boxplot(df, model_all_filter, all_filter, rename_dict,
                'All '+perfname+' Comparison','all_'+perfname.lower()+'_comparison','Model',perfname,False,False)
        
        #Heart plot
        model_heart_filter = ['Single NoSeg','Heart NoSeg','Triple NoSeg','Single WithSeg','Heart WithSeg','Triple WithSeg']
        heart_filter = ['h_','heart']
        clean_df_and_make_boxplot(df, model_heart_filter, heart_filter, rename_dict,
                'Heart '+perfname+' Comparison','heart_'+perfname.lower()+'_comparison','Model',perfname,False,False)
        
        #Left lung plot
        model_ll_filter = ['Single NoSeg','LLung NoSeg','Triple NoSeg','Single WithSeg','LLung WithSeg','Triple WithSeg']
        ll_filter = ['left_lung']
        clean_df_and_make_boxplot(df, model_ll_filter, ll_filter, rename_dict,
                'Left Lung '+perfname+' Comparison','leftlung_'+perfname.lower()+'_comparison','Model',perfname,False,False)
        
        #Right lung plot
        model_rl_filter = ['Single NoSeg','RLung NoSeg','Triple NoSeg','Single WithSeg','RLung WithSeg','Triple WithSeg']
        rl_filter = ['right_lung']
        clean_df_and_make_boxplot(df, model_rl_filter, rl_filter, rename_dict,
                'Right Lung '+perfname+' Comparison','rightlung_'+perfname.lower()+'_comparison','Model',perfname,False,False)
        
    
    
    