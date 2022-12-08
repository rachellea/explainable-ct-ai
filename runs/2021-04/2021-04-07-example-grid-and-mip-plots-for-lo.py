#2021-04-07-example-grid-and-mip-plots-for-lo.py

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
import pandas as pd

from src.deid import plot_grid

if __name__=='__main__':
    outbox_dir = '/storage/rlb61-outbox/2021-03-26-RAD-ChestCT-Honest-Broker'
    grid_plot_dir = '/home/rlb61/data/img-hiermodel2/results/results_2021/2021-04-07-Example-Grid-and-MIP-Plots-for-Lo'
    if not os.path.exists(grid_plot_dir):
        os.mkdir(grid_plot_dir)
    
    ct_scans_list = [x for x in os.listdir(outbox_dir) if '.npz' in x]
    
    #Make visualizations
    print('Making visualizations')
    for ctvol_filename in ct_scans_list[0:5]:
        plot_grid.make_grid_plot(ctvol_filename=ctvol_filename,
                                 ctvol_directory=outbox_dir,
                                 grid_plot_dir=grid_plot_dir)
    