import torch
import PIL
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torchvision
from torchvision import datasets, models, transforms
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
import torch
import torch.nn.functional as F
import torch.nn as nn

class FaceDataset(Dataset):
    def __init__(self, path2data, transform, transform_params=None):
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
        
        if self.transform_params:
            img, label_tensor = self.transform(img, label_tensor, self.transform_params)
        else:
            img, label_tensor = self.transform(img, label_tensor)

        return img, label_tensor
    
def set_parameter_requires_grad(model, feature_extracting):
    """This codes is inherited from pytorch tutorial.
    Original link :https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
def get_list_backbones():
    backbones = ["Mobilenet_v2"]
    backbones.append("Restnet18")
    backbones.append("Alexnet")
    backbones.append("VGG11_bn")
    backbones.append("Squeezenet1_0")
    backbones.append("Densenet121")
    return backbones
    
def initialize_backbone(backbone_name, feature_extract=False, use_pretrained=True):
    """This codes is inherited from pytorch tutorial.
    Original link :https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    
    """
    
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    backbone = None
    input_size = 0
    num_features = 0
    model_name = backbone_name
    
    if model_name == "Resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_features = model_ft.fc.in_features
        backbone = nn.Sequential(*list(model_ft.children())[:-1])
        input_size = 224
    
    elif model_name == "Mobilenet_v2":
        """ Alexnet
        """
        model_ft = models.mobilenet_v2(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_features = model_ft.classifier[6].in_features
        backbone = nn.Sequential(*list(model_ft.children())[:-1])
        backbone = nn.Sequential(
            backbone,
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )
        input_size = 224
    
    elif model_name == "Alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_features = 256
        backbone = nn.Sequential(*list(model_ft.children())[:-2])
        backbone = nn.Sequential(
            backbone,
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )
          
        input_size = 224

    elif model_name == "VGG11_bn":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_features = 512
        backbone = nn.Sequential(*list(model_ft.children())[:-2])
        backbone = nn.Sequential(
            backbone,
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )
        input_size = 224

    elif model_name == "Squeezenet1_0":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_features = 512
        backbone = nn.Sequential(*list(model_ft.children())[:-1])
        
        backbone = nn.Sequential(
            backbone,
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )
        
        input_size = 224

    elif model_name == "Densenet121":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_features = 160
        backbone = nn.Sequential(*list(model_ft.children())[:-1])
        
        backbone = nn.Sequential(
            backbone,
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )
        
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return backbone, input_size, num_features
        