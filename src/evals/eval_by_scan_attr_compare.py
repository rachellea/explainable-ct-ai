#eval_by_scan_attr_compare.py
#Copyright (c) 2021 Rachel Lea Ballantyne Draelos

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

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.ioff()

import seaborn

def compare_abs_auroc_differences(results_dir,
                                  model_a_path, model_a_descriptor,
                                  model_b_path, model_b_descriptor):
    """Take the results of eval_by_scan_attr.calculate_model_perf_by_scan_attr()
    for two different models and make comparison plots.
    
    <results_dir> is the path to the directory in which to save the results
    <model_a_path> is the path to the directory in which the results of
        eval_by_scan_attr.py are stored for Model A
    <model_b_path> is the path to the directory in which the results of
        eval_by_scan_attr.py are stored for Model B
    <model_a_descriptor> and <model_b_descriptor> are descriptive strings
        that will be used in generating the plots
    
    For each scan attribute, a df will be loaded that has the following format:
        the columns are different scan attribute options. For example if the
            attribute is StationName, then the options (the columns) can
            include 'DMPRAD3FORCE', 'DMP_CT1', 'DMP_CT2', 'CCCT3Revo',
            'IPCT1', 'CaryCT1',...,'CTC1'.
        the rows are different abnormalities, such as 'lung_nodule',
            'heart_cardiomegaly', and 'h_great_vessel_atherosclerosis'
        the values are AUROCs calculated for that particular scan attribute
            option and abnormality."""
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    
    #Create plots for all the attributes
    for attribute in ['SliceThickness','PatientAgeYears',
        'orig_square','orig_numslices','orig_slope','orig_inter',
        'orig_yxspacing','orig_zdiff',
        
        'Manufacturer','ManufacturerModelName','InstitutionName',
        'StationName','SoftwareVersions','ConvolutionKernel',
        'PatientSex','EthnicGroup','IterativeReconAnnotation',
        'IterativeReconConfiguration','IterativeReconLevel',
        'ProtocolName','ReconAlgo','ReconAlgoManuf']:
        model_a_df = pd.read_csv(os.path.join(model_a_path, attribute+'_AUROC.csv'),header=0,index_col=0)
        model_b_df = pd.read_csv(os.path.join(model_b_path,attribute+'_AUROC.csv'),header=0,index_col=0)
        
        model_a_df_w_diff = add_diff_column(model_a_df,model_a_descriptor)
        model_b_df_w_diff = add_diff_column(model_b_df,model_b_descriptor)
        
        #Combine the dataframes
        combined = pd.concat([model_a_df_w_diff, model_b_df_w_diff],axis=0, ignore_index=True)
    
        #sort by model and then by difference
        combined = combined.sort_values(by=['Model','Max AUROC Difference'],ascending=[False,True])
        
        #make plots
        make_bar_plot_per_abnormality(combined, attribute, results_dir)
        make_boxplot_agg_abnormalities(combined, attribute, results_dir)
        make_boxplot_agg_abnormality_groups(combined, attribute, results_dir)
    

def add_diff_column(df, model_descriptor):
    """Calculate the maximum AUROC difference between the different scan
    attribute options for each abnormality, and reformat the df for subsequent
    seaborn plotting.
    
    A bigger Max AUROC Difference is worse, because it means the performance
    varies a lot by that scan attribute.
    
    A smaller difference is good, because it means the performance for
    that abnormality is consistent across the different scan attribute
    options.
    
    For example, if the AUROC difference is very large for 'cardiomegaly'
    depending on different StationName (CT scanner) attribute options, that
    suggests the model might be cheating and using information about the
    CT scanner to predict 'cardiomegly.'
    
    <df> is a pandas dataframe with the format described in the
        docstring for compare_abs_auroc_differences()
    <model_descriptor> is a descriptive string"""
    df['Maximum'] = df.max(axis=1)
    df['Minimum'] = df.min(axis=1)
    df['Max AUROC Difference'] = (df['Maximum']-df['Minimum'])
    
    #drop 'Count' row
    df = df.drop(index='Count')
    
    #add a column indicating the model so you can use seaborn barplot easily
    df['Model'] = model_descriptor
    
    #make the abnormality index into a column so you can use seaborn barplot easily
    df.reset_index(inplace=True)
    df = df.rename(columns = {'index':'Abnormality'})
    
    #keep only the 3 columns needed for plotting
    df = df[['Abnormality','Max AUROC Difference','Model']]
    
    return df

def make_bar_plot_per_abnormality(combined, attribute, results_dir):
    """Make bar plot where each abnormality has two bars, one bar for Model A
    and one bar for Model B. The y axis shows the Max AUROC Difference, so
    lower is better."""
    fig, ax = plt.subplots(figsize=(16,8))
    seaborn.barplot(x = 'Abnormality', y = 'Max AUROC Difference', data = combined, 
                hue = 'Model', hue_order = ['Base','Mask'], ax = ax)
    plt.xticks(rotation=90, fontsize='x-small')
    plt.savefig(os.path.join(results_dir,attribute+'_BarPerAbn.png'))
    plt.close()
    
def make_boxplot_agg_abnormalities(combined, attribute, results_dir):
    """Boxplot where different abnormalities are aggregated for each model,
    and the y axis shows the Max AUROC Difference, so a lower overall
    boxplot is better"""
    fig, ax = plt.subplots(figsize=(6,6))
    seaborn.boxplot(x = 'Model', y = 'Max AUROC Difference', data = combined, ax = ax, order=['Base','Mask'])
    plt.title('Max AUROC Difference \nAcross Abnormalities',fontsize='xx-large')
    increase_label_sizes(plt)
    plt.savefig(os.path.join(results_dir,attribute+'_BoxAggAbns.png'))
    plt.close()

def make_boxplot_agg_abnormality_groups(combined, attribute, results_dir):
    """Grouped boxplot where different abnormalities are aggregated for each
    model, but abnormalities are split up according to their organ: lung,
    heart, great_vessel, or mediastinum. The y axis shows the Max AUROC
    Difference, so a lower overall boxplot is better."""
    #Assign an organ to each abnormality
    combined['Organ']=''
    for idx in combined.index.values.tolist():
        abnormality = combined.at[idx,'Abnormality']
        if 'lung' in abnormality:
            combined.at[idx,'Organ'] = 'lung'
        elif 'heart' in abnormality:
            combined.at[idx,'Organ'] = 'heart'
        elif 'vessel' in abnormality:
            combined.at[idx,'Organ'] = 'great_vessel'
        elif 'mediastinum' in abnormality:
            combined.at[idx,'Organ'] = 'mediastinum'
    
    #Sanity check: make sure every abnormality has an organ assigned
    assert combined[combined['Organ']==''].shape[0]==0
    
    #Make plot
    fig, ax = plt.subplots(figsize=(8,8))
    seaborn.boxplot(x = 'Model', y = 'Max AUROC Difference', order = ['Base','Mask'], 
                hue = 'Organ', data = combined, ax = ax, palette = 'mako')
    plt.title('Max AUROC Difference\nAcross Grouped Abnormalities',fontsize='xx-large')
    increase_label_sizes(plt)
    plt.savefig(os.path.join(results_dir,attribute+'_BoxAggAbnsByOrgan.png'))
    plt.close()

def increase_label_sizes(plt):
    """Increase the axis label sizes for a seaborn plot"""
    #https://stackoverflow.com/questions/43670164/font-size-of-axis-labels-in-seaborn?rq=1
    for ax in plt.gcf().axes:
        current_xlabels = ax.get_xlabel()
        ax.set_xlabel(current_xlabels, fontsize='x-large')
        current_ylabels = ax.get_ylabel()
        ax.set_ylabel(current_ylabels, fontsize='x-large')

    plt.xticks(fontsize='x-large')
    plt.yticks(fontsize='x-large')