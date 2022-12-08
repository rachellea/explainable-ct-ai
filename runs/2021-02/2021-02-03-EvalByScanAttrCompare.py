#2021-02-03-EvalByScanAttrCompare.py
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

from src.evals import eval_by_scan_attr_compare

if __name__ == '__main__':
    eval_by_scan_attr_compare.compare_abs_auroc_differences(
        results_dir = '/home/rlb61/data/img-hiermodel2/results/results_2021/2021-02-03_EvalByScanAttrCompare_Mask_vs_Base',
        model_a_path='/home/rlb61/data/img-hiermodel2/results/results_2021/2021-01-12_EvalByScanAttr-2020-10-09_WHOLEDATA_BodyAvg_Baseline-Test',
        model_a_descriptor = 'Base',
        model_b_path='/home/rlb61/data/img-hiermodel2/results/results_2021/2021-01-12_EvalByScanAttr-2020-10-08_WHOLEDATA_BodyAvg_Mask-Test/',
        model_b_descriptor = 'Mask')