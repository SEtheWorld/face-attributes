import torch
import torch.nn as nn
import PIL
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from shapely.geometry import Polygon
from math import pi
import torchvision
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
import time
from tqdm.notebook import tqdm

class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n):
        self.sum += val
        self.count += n

    def get_avg(self):
        return self.sum / self.count

class Pipeline:
    def __init__(self, model, train_dl, val_dl, performance, params):
        self.model = model
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.performance = performance
        self.params = params       

    def get_lr(self): 
        for param_group in self.params.opt.param_groups:
            return param_group['lr']

    def _process_epoch(self, training):
        loss_monitor = AverageMeter()
        metrics_monitor = AverageMeter()
        if training:
            dataset_dl = self.train_dl
        else:
            dataset_dl = self.val_dl

        for xb, yb in tqdm(dataset_dl):
            yb=yb.to(self.params.device)

            # get model output
            predictions = self.model(xb.to(self.params.device))

            # get loss per batch
            loss = self.performance.loss_function(predictions, yb)
            loss_monitor.update(loss, len(yb))

            # get metrcis per batch
            metrics = self.performance.metrics_function(predictions, yb)
            metrics_monitor.update(metrics, len(yb))

            # update parameters
            if training:
                self.params.opt.zero_grad()
                loss.backward()
                self.params.opt.step()

            if self.params.sanity_check:
                break
        return loss_monitor, metrics_monitor

    def train_val(self):
        # a deep copy of weights for the best performing model
        best_model_wts = copy.deepcopy(self.model.state_dict())
    
        # initialize best loss to a large value
        best_loss=float('inf')    
        num_epochs = self.params.num_epochs
        for epoch in range(num_epochs):
            # get current learning rate
            current_lr = self.get_lr()
            print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, current_lr))   
            # train the model
            self.model.train()

            t = time.time()
            train_loss_monitor, train_metrics_monitor = self._process_epoch(training = True)       
            train_time = (time.time() - t) / 60
            self.performance.log(train_loss_monitor, train_metrics_monitor, training = True, epoch_number=epoch)
            
            # evaluate the model
            self.model.eval()
            with torch.no_grad():
                t = time.time()
                val_loss_monitor, val_metrics_monitor = self._process_epoch(training = False,)
                val_time = (time.time() - t) / 60
                self.performance.log(val_loss_monitor, val_metrics_monitor, training = False, epoch_number= epoch)
            
            val_loss = val_loss_monitor.get_avg().item()
            train_loss = train_loss_monitor.get_avg().item()
            # store best model
            if  val_loss< best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
            
                # store weights into a local file
                torch.save(self.model.state_dict(), self.params.path2weights)
                print("Copied best model weights!")
            
            # learning rate schedule
            self.params.lr_scheduler.step(val_loss)
            if current_lr != self.get_lr():
                print("Loading best model weights!")
                self.model.load_state_dict(best_model_wts) 
        
            print("train loss: %.6f time : %.5f" %(train_loss, train_time))
            print("val loss: %.6f time: %.5f" %(val_loss, val_time))
            print("-"*10) 

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        return self.model, self.performance

    def test(self, test_dl):          
        # evaluate the model
        loss_monitor = AverageMeter()
        metrics_monitor = AverageMeter()
        self.model.eval()

        with torch.no_grad():
            for xb, yb in tqmd(test_dl):
                yb=yb.to(self.params.device)

                # get model output
                predictions = self.model(xb.to(self.params.device))

                # get loss per batch
                loss = self.performance.loss_function(predictions, yb)
                loss_monitor.update(loss, len(yb))

                # get metrcis per batch
                metrics = self.performance.metrics_function(predictions, yb)
                metrics_monitor.update(metrics, len(yb))

        return loss_monitor.get_avg(), metrics_monitor.get_avg()           
    
class Prams:
    def __init__(self, num_epochs, path2weights, device, optimizer, lr_scheduler, sanity_check):
        self.num_epochs = num_epochs
        self.path2weights = path2weights
        self.device = device
        self.opt = optimizer
        self.lr_scheduler = lr_scheduler
        self.sanity_check = sanity_check

class Performance:
    def __init__(self):
        # history of loss values in each epoch
        self.loss_history={
            "train": [],
            "val": [],
        }

        # histroy of metric values in each epoch
        self.metrics_history={
            "train": [],
            "val": [],
        }
        self.loss_func = nn.MSELoss(reduction="sum")
    def loss_function(self, predictions, targets):
        loss = self.loss_func(predictions, targets[:, 0].unsqueeze(1))      
        return loss

    def metrics_function(self, predictions, targets):
        mae = torch.abs(predictions - targets[:, 0].unsqueeze(1)).sum()      
        return mae
        
    def log(self, loss_monitor, metrics_monitor, training, epoch_number):
        if training:
            train_val = "train"
        else:
            train_val = "val"

        self.loss_history[train_val].append(loss_monitor.get_avg())
        self.metrics_history[train_val].append(metrics_monitor.get_avg())



