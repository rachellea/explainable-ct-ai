#eval_by_scan_attr_plot.py
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

from src.evals.eval_by_scan_attr_compare import increase_label_sizes 

def compare_auroc_across_attr_options(results_dir, model_path, model_descriptor):
    """Take the results of eval_by_scan_attr.calculate_model_perf_by_scan_attr()
    for one model and make plots comparing the AUROCs across different
    attribute options.
    See eval_by_scan_attr_compare.compare_abs_auroc_differences() for
    documentation."""
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
        df = pd.read_csv(os.path.join(model_path, attribute+'_AUROC.csv'),header=0,index_col=0)
        
        #Drop th Count row
        df = df.drop(index='Count')
        
        #Rename the columns that are too long (some of them are aggregations of
        #a lot of options and are extremely long)
        too_long_count = 0
        for colname in df.columns.values.tolist():
            if len(colname) > 30:
                newcolname = colname[0:20]+'etc'+str(too_long_count)
                df = df.rename(columns = {colname:newcolname})
                print('for',attribute,'renamed',colname,'to',newcolname)
                too_long_count+=1
        
        #Calculate the rank order of the options by median AUROC
        #so that the plot can be ordered by median AUROC. You need to do this
        #after renaming the columns otherwise you will have to rename the
        #columns within median_order too.
        medians = df.median(axis=0).sort_values(ascending=False)
        median_order = medians.index.values.tolist()
        
        #Make Abnormality one of the columns
        df = df.reset_index()
        df = df.rename(columns = {'index':'Abnormality'})
        
        #Melt the df
        value_vars = df.columns.values.tolist()
        value_vars.remove('Abnormality')
        #print(value_vars)
        melted=pd.melt(df, id_vars=['Abnormality'], value_vars=value_vars)
        melted = melted.rename(columns = {'variable':attribute,'value':'AUROC'})
        
        #Create plot and save
        fig, ax = plt.subplots(figsize=(6,6))
        seaborn.boxplot(x = attribute, y = 'AUROC', data = melted, ax = ax, order = median_order)
        increase_label_sizes(plt)
        plt.xticks(rotation=90, fontsize='small')
        plt.title('AUROC by '+attribute+'\nAggregated Across Abnormalities',fontsize='xx-large')
        plt.savefig(os.path.join(results_dir,model_descriptor+'_'+attribute+'_BoxByAttrOptionAggOverAbns.png'), bbox_inches='tight')
        plt.close()            
            