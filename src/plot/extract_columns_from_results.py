#extract_columns_from_results.py
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

def extract_columns(chosen_cols, keyword):
    files = {'accuracy':'valid_accuracy_all.csv',
             'auroc':'valid_auroc_all.csv',
             'avg_precision':'valid_avg_precision_all.csv',
             'top_k':'valid_top_k_all.csv'}
    for key in files:
        filename = files[key]
        x = pd.read_csv(filename,header=0,index_col=0)
        selected = x[chosen_cols]
        selected = selected.dropna(axis=0,how='all')
        selected.to_csv(keyword+'_'+key+'.csv',header=True,index=True)

def extract_columns_test_set(chosen_cols, keyword):
    files = {'accuracy':'test_accuracy_all.csv',
             'auroc':'test_auroc_all.csv',
             'avg_precision':'test_avg_precision_all.csv',
             'top_k':'test_top_k_all.csv'}
    for key in files:
        filename = files[key]
        x = pd.read_csv(filename,header=0,index_col=0)
        selected = x[chosen_cols]
        selected = selected.dropna(axis=0,how='all')
        selected.to_csv(keyword+'_'+key+'.csv',header=True,index=True)