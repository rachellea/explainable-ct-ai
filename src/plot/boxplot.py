#boxplot.py
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

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

"""Make a boxplot from <df> where the columns of <df> are different models,
the rows are different abnormalities, and the values are a performance
metric like AUROC. <rename_dict> is optional; if it is not empty then it
should map from the current column names to new column names and it will
be used to rename the columns before making the figure."""

def make_boxplot(df, rename_dict, plot_title, save_name, xlab, ylab, rotate_xticks, big):
    clean = df.rename(columns=rename_dict)
    ax = sns.boxplot(data = clean)
    plt.title(plot_title)
    plt.xticks(fontsize=8) #I think the default is 10
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
    plt.savefig(save_name+'.pdf')
