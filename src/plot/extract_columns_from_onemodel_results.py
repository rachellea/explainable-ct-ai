#extract_columns_from_onemodel_results.py
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

import pandas as pd

#extract columns from local results

def extract_valid_columns(prefix, epoch):
    #e.g. prefix is 'resnet18-interp-bigdata_'
    #e.g. epoch is 'epoch_17' (a string, the column name)
    files = {'accuracy':'valid_accuracy_Table.csv',
             'auroc':'valid_auroc_Table.csv',
             'avg_precision':'valid_avg_precision_Table.csv',
             'top_k':'valid_top_k_Table.csv'}
    for key in files:
        filename = files[key]
        x = pd.read_csv(prefix+filename,header=0,index_col=0)
        selected = x[epoch]
        selected.to_csv('valid_'+epoch+'_selected_'+key+'.csv',header=True,index=True)

def extract_test_columns(prefix, epoch):
    files = {'accuracy':'test_accuracy_Table.csv',
             'auroc':'test_auroc_Table.csv',
             'avg_precision':'test_avg_precision_Table.csv',
             'top_k':'test_top_k_Table.csv'}
    for key in files:
        filename = files[key]
        x = pd.read_csv(prefix+filename,header=0,index_col=0)
        selected = x[epoch]
        selected.to_csv('test_'+epoch+'_selected_'+key+'.csv',header=True,index=True)