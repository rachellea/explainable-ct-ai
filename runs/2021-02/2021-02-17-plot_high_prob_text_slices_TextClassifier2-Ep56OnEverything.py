#2021-02-17-plot_high_prob_text_slices_TextClassifier2-Ep56OnEverything.py
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

from src.plot import plot_high_prob_text_slices as phpt

if __name__ == '__main__':
    prefix = '/home/rlb61/data/img-hiermodel2/results/results_2021/'
    for directory in [
            '2021-02-16_TextClassifier2_Cube_FullAug_DeployEp56OnTest_axial',
            '2021-02-16_TextClassifier2_Cube_FullAug_DeployEp56OnTest_coronal',
            '2021-02-16_TextClassifier2_Cube_FullAug_DeployEp56OnTest_sagittal',
        
            '2021-02-11_TextClassifier2_Cube_FullAug_DeployEp56OnValid_axial',
            '2021-02-12_TextClassifier2_Cube_FullAug_DeployEp56OnValid_coronal',
            '2021-02-12_TextClassifier2_Cube_FullAug_DeployEp56OnValid_sagittal',
            
            '2021-02-15_TextClassifier2_Cube_FullAug_DeployEp56OnTrain_axial',
            '2021-02-16_TextClassifier2_Cube_FullAug_DeployEp56OnTrain_coronal',
            '2021-02-16_TextClassifier2_Cube_FullAug_DeployEp56OnTrain_sagittal'
            ]:
        results_dir = os.path.join(prefix,directory)
        phpt.plot_prob_histogram(results_dir, epoch=56)
        phpt.make_plot_of_high_prob_text_slices(results_dir=results_dir,
                            epoch=56,
                            ct_scan_path='/storage/rlb61-ct_images/vna/rlb61/2019-10-BigData-DEID/',
                            lower_thresh = 0.95,
                            upper_thresh = 1.0)
    
