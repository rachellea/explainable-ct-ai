#run_experiment.py
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

import gc
import os
import sys
import timeit
import psutil
import datetime
import numpy as np
import pandas as pd

import torch, torch.nn as nn
from torch.utils.data import DataLoader

from src.evals import evaluate, losses

#Set seeds
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

class DukeCTExperiment(object):
    def __init__(self, descriptor, base_results_dir,
                 custom_net, custom_net_args,
                 loss_string, loss_args, learning_rate, weight_decay,
                 num_epochs, patience, batch_size, device,
                 data_parallel, model_parallel, use_test_set, task,
                 old_params_dir, dataset_class, dataset_args):
        """Variables:
        <descriptor>: string describing the experiment. This descriptor will
            become part of a directory name, so it's a good idea not to
            include spaces in this string.
        <base_results_dir>: the path to the results directory. A new directory
            will be created within this directory to store the results of
            this experiment.
        <custom_net>: class defining a model. This class must inherit from
            nn.Module.
        <custom_net_args>: dictionary where keys correspond to custom net
            input arguments, and values are the desired values.
        <loss_string>: Defines the loss that will be used to train the model.
            e.g. 'bce' for binary cross entropy. All available loss strings
            are defined in losses.py.
        <loss_args>: dict of arguments for the loss function. Often this is
            empty.
        <learning_rate>: number, the learning rate that will be passed to
            the optimizer
        <weight_decay>: number, the weight decay that will be passed to the
            optimizer.
        <num_epochs>: int for the maximum number of epochs to train.
        <patience>: int for the number of epochs for which the validation set
            loss must fail to decrease in order to cause early stopping.
        <batch_size>: int for number of examples per batch
        <device>: int specifying which device to use
        <data_parallel>: if True then parallelize data across all GPUs.
        <model_parallel>: if True then distribute model weights across GPUs.
            At the moment you cannot have both data_parallel and model_parallel
            simultaneously.
        <use_test_set>: if True, then run model on the test set. If False, use
            only the training and validation sets. This is meant as an extra
            precaution against accidentally running anything on the test set.
        <task>:
            'train_eval': train and evaluate a new model.
                If <use_test_set> is False, then this will train and evaluate
                a model using only the training set and validation set,
                respectively.
                If <use_test_set> is True, then additionally the test set
                performance will be calculated for the best validation epoch.
            'restart_train_eval': restart training and evaluation of a model
                that wasn't done training (e.g. a model that died accidentally)
            'predict_on_train': load a trained model and make predictions on
                the training set using that model.
            'predict_on_valid': load a trained model and make predictions on
                the validation set using that model.
            'predict_on_test': load a trained model and make predictions on
                the test set using that model.
        <old_params_dir>: this is only needed if <task> is 'restart_train_eval',
            'predict_on_valid', or 'predict_on_test.' This is the path to the
            parameters that will be loaded in to the model.
        <dataset_class>: Dataset class that inherits from
            torch.utils.data.Dataset.
        <dataset_args>: dict of args to be passed to the <chosen_dataset>
            class"""
        self.descriptor = descriptor
        assert isinstance(descriptor,str)
        print(self.descriptor)
        self.base_results_dir = base_results_dir
        self.custom_net = custom_net
        self.custom_net_args = custom_net_args
        assert isinstance(self.custom_net_args,dict)
        if dataset_args['use_projections9']:
            self.custom_net_args['slices'] = 15
        else:
            self.custom_net_args['slices'] = 135
        self.loss_string = loss_string
        assert isinstance(self.loss_string,str)
        self.loss_args = loss_args
        assert isinstance(self.loss_args,dict)
        print('Using loss:',self.loss_string)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        assert isinstance(self.num_epochs,int)
        self.batch_size = batch_size
        assert isinstance(self.batch_size,int)
        print('self.batch_size=',self.batch_size)
        if 'mask' in loss_string.lower():
            assert self.batch_size==1, 'Batch size must be one when using a mask loss'
        #num_workers is number of threads to use for data loading
        self.num_workers = int(batch_size*16) #batch_size 1 = num_workers 16.
        print('self.num_workers=',self.num_workers)
        if self.num_workers == 1:
            print('Warning: Using only one worker will slow down data loading')
        
        #Set Device and Data Parallelism
        self.device = device
        assert self.device in [0,1,2,3,'all']
        self.data_parallel = data_parallel
        self.model_parallel = model_parallel
        self.deal_with_device_and_parallelism()
        
        #Set Task
        self.use_test_set = use_test_set
        assert isinstance(self.use_test_set,bool)
        self.task = task
        assert self.task in ['train_eval','restart_train_eval','predict_on_train','predict_on_valid','predict_on_test']
        if self.task in ['restart_train_eval','predict_on_train','predict_on_valid','predict_on_test']:
            self.load_params_dir = old_params_dir
        
        #Data and Labels
        self.CTDatasetClass = dataset_class
        self.dataset_args = dataset_args
        assert isinstance(self.dataset_args,dict)
        self.set_up_results_dirs() #Results dirs for output files and saved models, based on descriptor and dataset_args
        self.dataset_args['verbose'] = False
        self.dataset_args['loss_string'] = self.loss_string
        if self.task in ['train_eval','restart_train_eval','predict_on_train','predict_on_valid']:
            self.dataset_train = self.CTDatasetClass(setname = 'train', **self.dataset_args)
            self.dataset_valid = self.CTDatasetClass(setname = 'valid', **self.dataset_args)
            self.label_meanings = self.dataset_valid.return_label_meanings()
        if self.use_test_set:
            self.dataset_test = self.CTDatasetClass(setname = 'test', **self.dataset_args)
            self.label_meanings = self.dataset_test.return_label_meanings()
        
        #Tracking losses and evaluation results
        if self.task in ['train_eval','predict_on_train','predict_on_valid','predict_on_test']:
            self.train_loss = np.zeros((self.num_epochs))
            self.valid_loss = np.zeros((self.num_epochs))
            self.eval_results_valid, self.eval_results_test = evaluate.initialize_evaluation_dfs(self.label_meanings, self.num_epochs)
        elif self.task in ['restart_train_eval']:
            base_old_results_path = os.path.split(os.path.split(old_params_dir)[0])[0]
            self.train_loss = np.load(os.path.join(base_old_results_path, 'train_loss.npy'))
            self.valid_loss = np.load(os.path.join(base_old_results_path, 'valid_loss.npy'))
            self.eval_results_valid, self.eval_results_test = evaluate.load_existing_evaluation_dfs(self.label_meanings, self.num_epochs, base_old_results_path, self.descriptor)
        
        #For early stopping
        self.initial_patience = patience
        self.patience_remaining = patience
        self.best_valid_epoch = 0
        self.min_val_loss = np.inf
        
        #Run everything
        self.run_model()
    
    ### Methods ###
    def deal_with_device_and_parallelism(self):
        """Ensure device, data_parallel, and model_parallel are configured in
        an allowed way"""
        if self.device in [0,1,2,3]: #i.e. if a GPU number was specified:
            self.device = torch.device('cuda:'+str(self.device))
        #Make sure that data parallelism and model parallelism are not
        #simultaneously requested because this is not implemented
        assert not (self.data_parallel and self.model_parallel), 'Cannot have data_parallel and model_parallel simultaneously'
        #Only use device = 'all' when parallelism is needed
        if (self.data_parallel or self.model_parallel):
            assert self.device == 'all'
        if self.device == 'all':
            assert (self.data_parallel or self.model_parallel)
        if (self.device == 'all' and self.data_parallel):
            self.device = torch.device('cuda') #make all devices visible
        if (self.device == 'all' and self.model_parallel):
            self.device = None
            #device will be handled specially in the model definition and in
            #the function that puts the data on the device
        print('using device',str(self.device))
    
    def set_up_results_dirs(self):
        self.results_dir = os.path.join(self.base_results_dir,datetime.datetime.today().strftime('%Y-%m-%d')+'_'+self.descriptor)
        if not os.path.isdir(self.results_dir):
            os.mkdir(self.results_dir)
        self.params_dir = os.path.join(self.results_dir,'params')
        if not os.path.isdir(self.params_dir):
            os.mkdir(self.params_dir)
        self.backup_dir = os.path.join(self.results_dir,'backup')
        if not os.path.isdir(self.backup_dir):
            os.mkdir(self.backup_dir)
        #Only used for mask models where we need to save the attn_gr_truth:
        #First, if an attn_storage_dir is already explicitely specified, then
        #use it:
        if 'attn_storage_dir' in self.dataset_args['attn_gr_truth_prep_args'].keys():
            assert os.path.exists(self.dataset_args['attn_gr_truth_prep_args']['attn_storage_dir'])
            print('Using prespecified full attn_storage_dir',self.dataset_args['attn_gr_truth_prep_args']['attn_storage_dir'])
        else:
            #It has not been specified, which means we need to make a fresh one:
            attn_storage_dir = os.path.join(self.results_dir,'attn_storage_temp')
            if not os.path.isdir(attn_storage_dir):
                os.mkdir(attn_storage_dir)
            self.dataset_args['attn_gr_truth_prep_args']['attn_storage_dir'] = attn_storage_dir
            print('Using new empty attn_storage_dir',self.dataset_args['attn_gr_truth_prep_args']['attn_storage_dir'])
    
    def run_model(self):
        if not (self.data_parallel or self.model_parallel):
            self.model = self.custom_net(**self.custom_net_args).to(self.device)
        elif self.data_parallel:
            self.model = nn.DataParallel(self.custom_net(**self.custom_net_args)).to(self.device)
        elif self.model_parallel: #model is already on devices
            self.model = self.custom_net(**self.custom_net_args)
        
        #optimizer: https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e
        momentum = 0.99
        print('Running with optimizer lr='+str(self.learning_rate)+', momentum='+str(round(momentum,2))+' and weight_decay='+str(self.weight_decay))
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.learning_rate, momentum=momentum, weight_decay=self.weight_decay)
        
        if self.task in ['restart_train_eval','predict_on_train','predict_on_valid','predict_on_test']:
            params_path = self.determine_params_path()
            check_point = torch.load(params_path)
            self.model.load_state_dict(check_point['params'])
            self.optimizer.load_state_dict(check_point['optimizer'])
            #For restart_train_eval:
            if self.task == 'restart_train_eval':
                start_epoch = self.best_valid_epoch+1
                print('Resuming training at epoch',start_epoch)
        else:
            start_epoch = 0
        
        if self.task in ['train_eval', 'restart_train_eval']:
            train_dataloader = DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers = self.num_workers)
            valid_dataloader = DataLoader(self.dataset_valid, batch_size=self.batch_size, shuffle=False, num_workers = self.num_workers)                
            for epoch in range(start_epoch, self.num_epochs): # loop over the dataset multiple times
                print('Epoch',epoch)
                t0 = timeit.default_timer()
                self.train(train_dataloader, epoch)
                self.valid(valid_dataloader, epoch)
                self.save_evals(epoch)
                if self.patience_remaining <= 0:
                    print('No more patience (',self.initial_patience,') left at epoch',epoch)
                    print('--> Implementing early stopping. Best epoch was:',self.best_valid_epoch)
                    break
                t1 = timeit.default_timer()
                print('Epoch',epoch,'time:',round((t1 - t0)/60.0,2),'minutes')
        if self.task=='predict_on_train': self.predict_on_train(DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=False, num_workers = self.num_workers), epoch=self.best_valid_epoch)
        if self.task=='predict_on_valid': self.valid(DataLoader(self.dataset_valid, batch_size=self.batch_size, shuffle=False, num_workers = self.num_workers), epoch=self.best_valid_epoch)
        if (self.task=='predict_on_test' and self.use_test_set): self.test(DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers = self.num_workers), epoch=self.best_valid_epoch)
        self.save_final_summary()
    
    def determine_params_path(self):
        """Determine what model to load based on what was provided in
        self.load_params_dir"""
        if os.path.isdir(self.load_params_dir):
            #A directory has been provided so we need to pick the latest model
            #that's stored within that directory
            all_saved_checkpoints = os.listdir(self.load_params_dir)
            all_epochs = [int(filename.split('epoch')[1]) for filename in all_saved_checkpoints]
            latest_epoch = max(all_epochs)
            self.best_valid_epoch = latest_epoch
            params_path = os.path.join(self.load_params_dir,self.descriptor+'_epoch'+str(latest_epoch))
        else:
            #A file of parameters has been provided directly, so we should use
            #that file
            assert os.path.isfile(self.load_params_dir)
            self.best_valid_epoch = int(self.load_params_dir.split('epoch')[1])
            params_path = self.load_params_dir
        print('For',self.task,'loading model params and optimizer state from params_path=',params_path)    
        return params_path
    
    def train(self, dataloader, epoch):
        self.model.train()
        epoch_loss, pred_epoch, gr_truth_epoch, volume_accs_epoch = self.iterate_through_batches(self.model, dataloader, epoch, training=True)
        self.train_loss[epoch] = epoch_loss
        self.plot_roc_and_pr_curves('train', epoch, pred_epoch, gr_truth_epoch)
        print("{:5s} {:<3d} {:11s} {:.3f}".format('Epoch', epoch, 'Train Loss', epoch_loss))
        
    def valid(self, dataloader, epoch):
        self.model.eval()
        with torch.no_grad():
            epoch_loss, pred_epoch, gr_truth_epoch, volume_accs_epoch = self.iterate_through_batches(self.model, dataloader, epoch, training=False)
        self.valid_loss[epoch] = epoch_loss
        self.eval_results_valid = evaluate.evaluate_all(self.eval_results_valid, epoch,
            self.label_meanings, gr_truth_epoch, pred_epoch)
        self.early_stopping_check(epoch, pred_epoch, gr_truth_epoch, volume_accs_epoch)
        print("{:5s} {:<3d} {:11s} {:.3f}".format('Epoch', epoch, 'Valid Loss', epoch_loss))
    
    def early_stopping_check(self, epoch, val_pred_epoch, val_gr_truth_epoch, val_volume_accs_epoch):
        """Check whether criteria for early stopping are met and update
        counters accordingly"""
        val_loss = self.valid_loss[epoch]
        if (val_loss < self.min_val_loss) or epoch==0: #then save parameters
            self.min_val_loss = val_loss
            check_point = {'params': self.model.state_dict(),                            
                           'optimizer': self.optimizer.state_dict()}
            torch.save(check_point, os.path.join(self.params_dir, self.descriptor+'_epoch'+str(epoch)))
            self.best_valid_epoch = epoch
            self.patience_remaining = self.initial_patience
            print('model saved, val loss',val_loss)
            self.plot_roc_and_pr_curves('valid', epoch, val_pred_epoch, val_gr_truth_epoch)
            self.save_all_pred_probs('valid', epoch, val_pred_epoch, val_gr_truth_epoch, val_volume_accs_epoch)
        else:
            self.patience_remaining -= 1
    
    def test(self, dataloader, epoch):
        self.model.eval()
        with torch.no_grad():
            epoch_loss, pred_epoch, gr_truth_epoch, volume_accs_epoch = self.iterate_through_batches(self.model, dataloader, epoch, training=False)
        self.eval_results_test = evaluate.evaluate_all(self.eval_results_test, epoch,
            self.label_meanings, gr_truth_epoch, pred_epoch)
        self.plot_roc_and_pr_curves('test', epoch, pred_epoch, gr_truth_epoch)
        self.save_all_pred_probs('test', epoch, pred_epoch, gr_truth_epoch, volume_accs_epoch)
        print("{:5s} {:<3d} {:11s} {:.3f}".format('Epoch', epoch, 'Test Loss', epoch_loss))
    
    def predict_on_train(self, dataloader, epoch):
        """This method is for saving predictions on the training set based on
        a saved model. This method has NOTHING to do with actually training
        a model."""
        self.model.eval()
        with torch.no_grad():
            epoch_loss, pred_epoch, gr_truth_epoch, volume_accs_epoch = self.iterate_through_batches(self.model, dataloader, epoch, training=False)
        self.save_all_pred_probs('train', epoch, pred_epoch, gr_truth_epoch, volume_accs_epoch)
        
    def iterate_through_batches(self, model, dataloader, epoch, training):
        epoch_loss = 0
        self.memory_logger_df = pd.DataFrame(index=[x for x in range(len(dataloader.dataset))], columns=['ObjectStringStart','ObjectCountStart','MaxGPUMemGiBStart',
                                                                                                         'ObjectStringEnd',  'ObjectCountEnd',  'MaxGPUMemGiBEnd'])
        #Initialize numpy arrays for storing results. examples x labels
        #Do NOT use concatenation, or else you will have memory fragmentation.
        num_examples = len(dataloader.dataset)
        num_labels = len(self.label_meanings)
        pred_epoch = np.zeros([num_examples,num_labels])
        gr_truth_epoch = np.zeros([num_examples,num_labels])
        volume_accs_epoch = np.empty(num_examples,dtype='U32') #need to use U32 to allow string of length 32
        
        for batch_idx, batch in enumerate(dataloader):
            self.update_memory_logger_df('Start',batch_idx,epoch,training)
            
            #Move data and ground truth to device
            data, gr_truth = self.move_data_to_device(batch)
            
            #Run the model on the data
            self.optimizer.zero_grad()
            if training:
                out = model(data)
            else:
                with torch.set_grad_enabled(False):
                   out = model(data)
            
            #Calculate loss
            if self.task == 'predict_on_test':
                train_labels_df = None
            else:
                train_labels_df = self.dataset_train.labels_df
            all_loss_args = {'loss_string':self.loss_string,
                'out':out,
                'gr_truth':gr_truth,
                'train_labels_df':train_labels_df,
                'device':self.device,
                'epoch':epoch,
                'training':training,
                'batch':batch,
                'loss_args':self.loss_args}
            loss = losses.calculate_loss(**all_loss_args)
            
            if training:
                loss.backward()
                self.optimizer.step()   
            
            epoch_loss += float(loss.item())
            del loss
            torch.cuda.empty_cache()
                        
            #Save predictions and ground truth across batches
            pred_np, gr_truth_np = losses.calculate_pred_np_and_gr_truth_np(self.loss_string, out, gr_truth, batch)
            start_row = batch_idx*self.batch_size
            stop_row = min(start_row + self.batch_size, num_examples)
            pred_epoch[start_row:stop_row,0:pred_np.shape[1]] = pred_np #pred_epoch is e.g. [25355,80] and pred_np is e.g. [1,80] for a batch size of 1
            gr_truth_epoch[start_row:stop_row,0:gr_truth_np.shape[1]] = gr_truth_np #gr_truth_epoch has same shape as pred_epoch
            volume_accs_epoch[start_row:stop_row] = batch['volume_acc'] #volume_accs_epoch stores the volume accessions in the order they were used
            
            #Cleanup
            del out
            del batch
            del data
            del gr_truth
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            gc.collect()
            #TODO MAYBE TRY: torch.cuda.memory_snapshot() for help with debugging
            self.update_memory_logger_df('End',batch_idx,epoch,training)
            
        return epoch_loss, pred_epoch, gr_truth_epoch, volume_accs_epoch    
    
    def move_data_to_device(self, batch):
        """Move data and ground truth to device. If data is for a three-channel
        model, then assume batch size of 1 and reshape so I can use a 2D
        feature extractor - i.e. so I can treat the volume like a batch of
        3-channel images."""
        #For model_parallel, 'You only need to make sure that the labels are
        #on the same device as the outputs when calling the loss function.'
        if self.model_parallel:
            gr_truth = batch['gr_truth'].to(torch.device('cuda:0'))#predictions end up on cuda:0
            #I'll take care of putting the data on the device inside the model
            return batch['data'], gr_truth

        #Move data and ground truth to device
        if self.dataset_args['view'] in ['axial','coronal','sagittal']:
            data = self._move_crop_type_data_to_device(croptypedata = batch['data'])
        elif self.dataset_args['view'] == 'all':
            data = {}
            data['axial'] = self._move_crop_type_data_to_device(croptypedata = batch['data']['axial'])
            data['sagittal'] = self._move_crop_type_data_to_device(croptypedata = batch['data']['sagittal'])
            data['coronal'] = self._move_crop_type_data_to_device(croptypedata = batch['data']['coronal'])
        
        #Ground truth to device
        if not self.dataset_args['volume_prep_args']['selfsupervised']:
            gr_truth = batch['gr_truth'].to(self.device)
            return data, gr_truth
        else: #if using selfsupervised learning, then the ground truth needs
            #to also contain the vector indicating what kind of data
            #augmentation was performed. dim=1 because first dimension is
            #the batch dimension
            gr_truth = torch.cat((batch['gr_truth'],batch['auglabel']),dim=1).to(self.device)
            return data, gr_truth
    
    def _move_crop_type_data_to_device(self, croptypedata):
        """If crop type is single, <croptypedata> is a tensor that can be
        moved directly to the device. If crop type is triple, <croptypedata>
        is a dictionary of tensors"""
        if self.dataset_args['volume_prep_args']['crop_type'] == 'single':
            return croptypedata.to(self.device)
        elif self.dataset_args['volume_prep_args']['crop_type'] == 'triple':
            data = {}
            data['heart'] = croptypedata['heart'].to(self.device)
            data['left_lung'] = croptypedata['left_lung'].to(self.device)
            data['right_lung'] = croptypedata['right_lung'].to(self.device)
            return data
    
    def plot_roc_and_pr_curves(self, setname, epoch, pred_epoch, gr_truth_epoch):
        outdir = os.path.join(self.results_dir,'curves')
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        evaluate.plot_roc_curve_multi_class(label_meanings=self.label_meanings,
                    y_test=gr_truth_epoch, y_score=pred_epoch,
                    outdir = outdir, setname = setname, epoch = epoch)
        evaluate.plot_pr_curve_multi_class(label_meanings=self.label_meanings,
                    y_test=gr_truth_epoch, y_score=pred_epoch,
                    outdir = outdir, setname = setname, epoch = epoch)
    
    def save_all_pred_probs(self, setname, epoch, pred_epoch, gr_truth_epoch, volume_accs_epoch):
        outdir = os.path.join(self.results_dir,'pred_probs')
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        (pd.DataFrame(pred_epoch,columns=self.label_meanings,index=volume_accs_epoch.tolist())).to_csv(os.path.join(outdir, setname+'_predprob_ep'+str(epoch)+'.csv'))
        (pd.DataFrame(gr_truth_epoch,columns=self.label_meanings,index=volume_accs_epoch.tolist())).to_csv(os.path.join(outdir, setname+'_grtruth_ep'+str(epoch)+'.csv'))
        
    def save_evals(self, epoch):
        if not self.use_test_set:
            evaluate.save(self.eval_results_valid, self.results_dir, self.descriptor+'_valid')
        else:
            evaluate.save(self.eval_results_test, self.results_dir, self.descriptor+'_test')
        evaluate.plot_learning_curves(self.train_loss, self.valid_loss, self.results_dir, self.descriptor)
        self.plot_last_layer_weights(epoch)
    
    def save_final_summary(self):
        if not self.use_test_set:
            evaluate.save_final_summary(self.eval_results_valid, self.best_valid_epoch, 'valid', self.results_dir)
        else:
            evaluate.save_final_summary(self.eval_results_test, self.best_valid_epoch, 'test', self.results_dir)
        evaluate.clean_up_output_files(self.best_valid_epoch, self.results_dir)

    def update_memory_logger_df(self,position,batch_idx,epoch,training):
        """Fill out <memory_logger_df> for this batch. The columns are:
          'ObjectStringStart','ObjectCountStart','MaxGPUMemGiBStart',
          'ObjectStringEnd','ObjectCountEnd','MaxGPUMemGiBEnd'
        """
        #Position specifies if we are at the Start of a batch or the End of a batch
        assert position in ['Start','End']
        #Memory logging.
        obj_count = memory_report()
        #self.memory_logger_df.at[batch_idx,'ObjectString'+position] = obj_string
        self.memory_logger_df.at[batch_idx,'ObjectCount'+position] =  obj_count
        
        #There are 1073741824 bytes in 1 GiB
        self.memory_logger_df.at[batch_idx,'MaxGPUMemGiB'+position] = round(torch.cuda.max_memory_allocated(device=self.device)/1073741824,3)
        if training:
            trainevalstr = 'train'
        else:
            trainevalstr = 'eval'
        self.memory_logger_df.to_csv(os.path.join(self.results_dir,'memory_logger_df_epoch'+str(epoch)+'_'+trainevalstr+'.csv'),header=True,index=True)
        if (batch_idx % 5000 == 0) and (position=='End'):
            print('\tEnd of batch',batch_idx)
        torch.cuda.reset_max_memory_cached(device=self.device)

#The following functions are from here:
#https://discuss.pytorch.org/t/how-pytorch-releases-variable-garbage/7277
def memory_report():
    #obj_string = ''
    obj_count = 0
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            #obj_string+=('[OBJECT: '+str(type(obj))+', '+str(obj.size())+', requires_grad='+str(obj.requires_grad)+', sum='+str(torch.sum(obj))+']')
            obj_count+=1
    #return obj_string, obj_count
    return obj_count

def get_cpu_statistics():
        print(sys.version)
        print(psutil.cpu_percent())
        print(psutil.virtual_memory())  # physical memory usage
        pid = os.getpid()
        py = psutil.Process(pid)
        memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
        print('memory GB:', memoryUse)

