#2021-01-12_EvalByScanAttr.py
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

from src.evals import eval_by_scan_attr

if __name__=='__main__':
    #AxialNet (BodyAvg) Mask model, validation set
    eval_by_scan_attr.run_all(
        descriptor = 'EvalByScanAttr-2020-10-08_WHOLEDATA_BodyAvg_Mask-Valid',
        grtruth_file = '/home/rlb61/data/img-hiermodel2/results/results_2019-2020/2020-10/2020-10-08_WHOLEDATA_BodyAvg_Mask_dilateFalse_nearest_Lambda33_FreshStart/pred_probs/valid_grtruth_ep4.csv',
        predprob_file = '/home/rlb61/data/img-hiermodel2/results/results_2019-2020/2020-10/2020-10-08_WHOLEDATA_BodyAvg_Mask_dilateFalse_nearest_Lambda33_FreshStart/pred_probs/valid_predprob_ep4.csv',
        phi_or_deid = 'phi')
    
    #AxialNet (BodyAvg) Mask model, test set
    eval_by_scan_attr.run_all(
        descriptor = 'EvalByScanAttr-2020-10-08_WHOLEDATA_BodyAvg_Mask-Test',
        grtruth_file = '/home/rlb61/data/img-hiermodel2/results/results_2019-2020/2020-10/2020-10-08_WHOLEDATA_BodyAvg_Mask_dilateFalse_nearest_Lambda33_FreshStart_TEST/pred_probs/test_grtruth_ep4.csv',
        predprob_file = '/home/rlb61/data/img-hiermodel2/results/results_2019-2020/2020-10/2020-10-08_WHOLEDATA_BodyAvg_Mask_dilateFalse_nearest_Lambda33_FreshStart_TEST/pred_probs/test_predprob_ep4.csv',
        phi_or_deid = 'phi')
    
    #AxialNet (BodyAvg) baseline model, test set
    eval_by_scan_attr.run_all(
        descriptor = 'EvalByScanAttr-2020-10-09_WHOLEDATA_BodyAvg_Baseline-Test',
        grtruth_file = '/home/rlb61/data/img-hiermodel2/results/results_2019-2020/2020-10/2020-10-09_WHOLEDATA_BodyAvg_Baseline_FreshStart_TEST/pred_probs/test_grtruth_ep4.csv',
        predprob_file = '/home/rlb61/data/img-hiermodel2/results/results_2019-2020/2020-10/2020-10-09_WHOLEDATA_BodyAvg_Baseline_FreshStart_TEST/pred_probs/test_predprob_ep4.csv',
        phi_or_deid = 'phi')