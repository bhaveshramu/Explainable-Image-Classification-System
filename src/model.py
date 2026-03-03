import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes):
    model = models.resnet18(pretrained=True)
    
    # Freeze earlier layers (lighter training)
    for name,param in model.named_parameters():
        if "layer4" not in name:
            param.requires_grad = False

    # Replace final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model