import torch

from load_data import load_data
from model.vgg16 import vgg16
from model.tiny import TinyClassifier2d
from model.resnet50 import resnet50
from train import train_model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print ('Using device:', device)

train_loader, validate_loader, _ = load_data()

# VGG16 Network
# vgg16 = vgg16()
# train_model(vgg16, 'vgg16', train_loader, validate_loader, 3, device, one_batch=True)

# Tiny Residual Network
# tiny = TinyClassifier2d()
# train_model(tiny, 'tiny', train_loader, validate_loader, 3, device, one_batch=True)

# ResNet50 Network
resnet50 = resnet50()
train_model(resnet50, 'resnet50', train_loader, validate_loader, 1, device, one_batch=True)
