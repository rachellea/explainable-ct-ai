#main_viz.py
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

import os
import torch
import datetime
import numpy as np
from torch.utils.data import DataLoader

#Set seeds
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

class MakeVisualizations(object):
    def __init__(self, descriptor, base_results_dir,
                 results_dir,
                 custom_net, custom_net_args,
                 batch_size, device,
                 full_params_dir, dataset_class, dataset_args):
        """<base_results_dir>: the base results directory. A new directory
            will be created within this directory to store the results of
            this experiment.
        <results_dir>: if None, then a new directory for the results of
            this experiment will be created within base_results_dir. If
            this is a path to an existing dir then all results from this
            experiment will be stored directly in this existing dir."""
        self.descriptor = descriptor
        self.custom_net = custom_net
        self.custom_net_args = custom_net_args
        self.batch_size = batch_size
        self.device = device
        self.full_params_dir = full_params_dir
        self.CTDatasetClass = dataset_class
        self.dataset_args = dataset_args
        
        #Setup
        print(self.descriptor)
        
        if results_dir is not None:
            self.results_dir = results_dir
        else:
            self.results_dir = os.path.join(base_results_dir,datetime.datetime.today().strftime('%Y-%m-%d')+'_'+self.descriptor)
        if not os.path.isdir(self.results_dir):
            os.mkdir(self.results_dir)
        
        self.num_workers = 16
        if self.device in [0,1,2,3]: #i.e. if a GPU number was specified:
            self.device = torch.device('cuda:'+str(self.device))
        else:
            assert False, 'invalid device specification'
        self.dataset_test = self.CTDatasetClass(setname = 'test', **self.dataset_args)
        self.label_meanings = self.dataset_test.return_label_meanings()
    
    def run_model(self):
        self.sigmoid = torch.nn.Sigmoid()
        test_dataloader = DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers = self.num_workers)
        self.viz_dataset(test_dataloader)
    
    def viz_dataset(self, dataloader):
        """This method will be overwritten by classes that inherit from
        this class"""
        pass        
    