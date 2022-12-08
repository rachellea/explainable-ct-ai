#2021-03-10-multipanel_group_plot_of_true_pos_w_highest_pred_prob.py

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

from src.attn_analysis import multipanel_group_plot

if __name__=='__main__':
    multipanel_group_plot.run_simple(results_dir='/home/rlb61/data/img-hiermodel2/results/results_2021/2021-03-11-MultiPanel-Group-Plots-TruePosHighPredProb-Trilinear/',
                              mask_dir='/home/rlb61/data/img-hiermodel2/results/results_2021/2021-03-10_Valid-Plot-True-Pos-with-Highest-PredProb-2020-10-08-Mask-trilinear',
                              base_dir='/home/rlb61/data/img-hiermodel2/results/results_2021/2021-03-10_Valid-Plot-True-Pos-with-Highest-PredProb-2020-10-09-Base-trilinear')
    
    multipanel_group_plot.run_simple(results_dir='/home/rlb61/data/img-hiermodel2/results/results_2021/2021-03-11-MultiPanel-Group-Plots-TruePosHighPredProb-NN/',
                              mask_dir='/home/rlb61/data/img-hiermodel2/results/results_2021/2021-03-10_Valid-Plot-True-Pos-with-Highest-PredProb-2020-10-08-Mask-nn',
                              base_dir='/home/rlb61/data/img-hiermodel2/results/results_2021/2021-03-10_Valid-Plot-True-Pos-with-Highest-PredProb-2020-10-09-Base-nn')
    
    
    
    
    



