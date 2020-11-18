import torch
import torch.nn 
import PIL
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torchvision
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy

class Face_dataset(Dataset):
    def __init__(self, path2data, transform, transform_params):
        self.df = pd.read_csv(path2data)  
        self.transform = transform
        self.transform_params = transform_params 

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        path = self.df.iloc[idx]["file_name"]
        age = self.df.iloc[idx]["age"].astype(np.float32)
        gender = self.df.iloc[idx]["gender"].astype(np.float32)

        img = Image.open(path)
        
        label_list = [age, gender]
        label_tensor = torch.tensor(label_list)

        img, label_tensor = self.transform(img, label_tensor, self.transform_params)

        return img, label_tensor
        
class AverageMeter:
    def __init__(self):
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n):
        self.sum += val
        self.count += n
    
    def get_avg(self):
        return (self.sum / self.count)

class FaceMetrics:
    def __init__(self):
        self.age_mae = AverageMeter()
        self.gender_acc = AverageMeter()

    def update(self, mae_val, acc_val, n):
        self.age_mae.update(mae_val, n)
        self.gender_acc.update(acc_val, n)
    




class Pipeline:
    def __init__(self, model, train_dl, val_dl, params, device):
        self.device = device
        self.opt = params["optimizer"]
        self.sanity_check = params["sanity_check"]
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.model = model
        self.num_epochs = params["num_epochs"]
        self.lr_scheduler = params["lr_scheduler"]
        self.path2weights = params["path2weights"]
        self.history = params["history"]
        self.loss_monitor = params["loss_monitor"]
        self.metrics_monitor = params["metrics_monitor"]

    def _process_batch(self, predictions, targets, loss_monitor, metrics_monitor, training):
        # get loss batch
        loss_monitor = loss_monitor.process_batch(predictions, targets)
        
        # get performance metric
        metrics_monitor = metrics_monitor.process_batch(predictions, targets)
        
        # update parameters
        if training:
            self.opt.zero_grad()
            loss_monitor.loss.backward()
            self.opt.step()
        
        return loss_monitor, metrics_monitor

    def _process_epoch(self, loss_monitor, metrics_monitor, training):

        if training:
            dataset_dl = self.train_dl
        else:
            dataset_dl = self.val_dl

        for xb, yb in dataset_dl:
            yb=yb.to(self.device)

            # get model output
            predictions = self.model(xb.to(self.device))

            # get loss per batch
            loss_monitor, metrics_monitor = self._process_batch(predictions, yb, loss_monitor, metrics_monitor, training)

            if sanity_check:
                break
        return loss_monitor, metrics_monitor

    def _get_lr(self): 
        for param_group in self.opt.param_groups:
            return param_group['lr']

    def train_val(self):
        # a deep copy of weights for the best performing model
        best_model_wts = copy.deepcopy(self.model.state_dict())
    
        # initialize best loss to a large value
        best_loss=float('inf')    
    
        for epoch in range(self.num_epochs):
            # get current learning rate
            current_lr = self._get_lr(opt)
            print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, current_lr))   

            # train the model
            self.model.train()
            self.loss_monitor.clear()
            self.metrics_monitor.clear()
            loss_monitor, metrics_monitor = self._process_epoch(self.loss_monitor, self.metrics_monitor, training=True)

            #collect loss and metric for training dataset
            self.history.update(loss_monitor, metrics_monitor, training=True)
            #metric_history["train"].append(train_metric)
        
            # evaluate the model
            model.eval()
            self.loss_monitor.clear()
            self.metrics_monitor.clear()
            with torch.no_grad():
                loss_monitor, metrics_monitor = self._process_epoch(self.loss_monitor, self.metrics_monitor, training=False)
       
            # collect loss and metric for validation dataset
            self.history.update(loss_monitor, metrics_monitor, training=False)
            
        
            # store best model
            if loss_monitor.loss.item() < best_loss:
                best_loss = loss_monitor.loss.item()
                best_model_wts = copy.deepcopy(model.state_dict())
            
                # store weights into a local file
                torch.save(model.state_dict(), path2weights)
                print("Copied best model weights!")
            
            # learning rate schedule
            lr_scheduler.step(self.monitor.loss.item())
            if current_lr != self._get_lr(opt):
                print("Loading best model weights!")
                model.load_state_dict(best_model_wts) 
            

        # load best model weights
        model.load_state_dict(best_model_wts)
        return self.model, self.history

