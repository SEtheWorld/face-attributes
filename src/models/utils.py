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

class FaceDataset(Dataset):
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
        