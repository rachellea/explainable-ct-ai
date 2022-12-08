#2021-04-13-random-3000-grid-and-mip-plots-of-radchestct.py

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

from src.deid import plot_grid

if __name__=='__main__':
    ctvol_directory = '/scratch/rlb61/2019-10-BigData-DEID/'
    
    #First select a random set of 3,000 CT volumes
    all_cts = os.listdir(ctvol_directory)
    assert len(all_cts)==36316
    assert len([x for x in all_cts if '.npz' in x])==36316
    random_3k = np.random.choice(all_cts,size=3000,replace=False).tolist()
    assert len(random_3k)==len(set(random_3k))
    
    #Visualize these CTs
    grid_plot_dir = '/home/rlb61/data/img-hiermodel2/results/results_2021/2021-04-13-Random-3000-Grid-and-MIP-Plots-of-RADChestCT'
    if not os.path.exists(grid_plot_dir):
        os.mkdir(grid_plot_dir)
        
    #Make visualizations
    print('Making visualizations')
    for idx, ctvol_filename in enumerate(random_3k):
        print(ctvol_filename,round(((idx*100)/3000), 2),'percent')
        plot_grid.make_grid_plot(ctvol_filename=ctvol_filename,
                                 ctvol_directory=ctvol_directory,
                                 grid_plot_dir=grid_plot_dir)
    