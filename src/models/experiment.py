import torch
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

class Experiment:
    def __init__(self, train_dl, val_dl, model, learning_procedure, device, sanity_check):
        self.model = model
        self.learning_procedure = learning_procedure
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.device = device
        self.sanity_check = sanity_check

    def run(self):
        pipeline = Pipeline(self.model, self.train_dl, self.val_dl, self.learning_procedure, self.device, self.sanity_check)
        model, performance = pipeline.train_val()
        return model, performance

class Pipeline:
    def __init__(self, model, train_dl, val_dl, learning_procedure, device, sanity_check):
        self.model = model
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.learning_procedure = learning_procedure
        self.device = device
        self.sanity_check = sanity_check

        params = self.learning_procedure.params       
        self.num_epochs = params.num_epochs
        self.path2weights = params.path2weights

        iteration = self.learning_procedure.iteration
        self.opt = iteration.opt
        self.lr_scheduler = iteration.lr_scheduler
        self.get_lr = iteration.get_lr

        self.performance = self.learning_procedure.performance
        self.loss_function = self.performance.loss_function
        self.metrics_function = self.performance.metrics_function
        

    def _process_epoch(self, training, epoch_number):

        if training:
            dataset_dl = self.train_dl
        else:
            dataset_dl = self.val_dl

        for xb, yb in dataset_dl:
            yb=yb.to(self.device)

            # get model output
            predictions = self.model(xb.to(self.device))

            # get loss per batch
            loss = self.loss_function(predictions, yb)
            # update parameters
            if training:
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            else:
                metrcis = self.metrics_function(predictions, yb)
            if self.sanity_check:
                break


    def train_val(self):
        # a deep copy of weights for the best performing model
        best_model_wts = copy.deepcopy(self.model.state_dict())
    
        # initialize best loss to a large value
        best_loss=float('inf')    
    
        for epoch in range(self.num_epochs):
            # get current learning rate
            current_lr = self.get_lr()
            print('Epoch {}/{}, current lr={}'.format(epoch, self.num_epochs - 1, current_lr))   

            # train the model
            self.model.train()
            self._process_epoch(training = True, epoch_number=epoch)
            self.performance.log(training = True)

            # evaluate the model
            self.model.eval()
            with torch.no_grad():
                self._process_epoch(training = False, epoch_number = epoch)
                self.performance.log(training = False)
            # store best model
            if self.performance.avg_loss < best_loss:
                best_loss = self.performance.avg_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
            
                # store weights into a local file
                torch.save(self.model.state_dict(), self.path2weights)
                print("Copied best model weights!")
            
            # learning rate schedule
            self.lr_scheduler.step(self.performance.avg_loss)
            if current_lr != self.get_lr():
                print("Loading best model weights!")
                self.model.load_state_dict(best_model_wts) 
            

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        return self.model, self.learning_procedure.performance
    
class LearningProcedure:
    def __init__(self, performance, iteration, params):
        self.iteration = iteration
        self.performance = performance
        self.params = params

class Prams:
    def __init__(self, num_epochs, path2weights):
        self.num_epochs = num_epochs
        self.path2weights = path2weights

class Iteration:
    def __init__(self, optimizer, lr_scheduler):
        self.opt = optimizer
        self.lr_scheduler = lr_scheduler

    def get_lr(self): 
        for param_group in self.opt.param_groups:
            return param_group['lr']

class Performance:
    def __init__():
        pass
    
    def loss_function(self, predictions, targets, training, epoch_number):
        pass

    def metrics_function(self, predictions, targets, training, epoch_number):
        pass
    

