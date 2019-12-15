import torch

from load_data import load_data
from model.vgg16 import vgg16
from model.tiny import TinyClassifier2d
from train import train_model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print ('Using device:', device)

train_loader, validate_loader, test_loader = load_data()

# VGG16 Network
# vgg16 = vgg16()
# train_model(vgg16, 'vgg16', train_loader, validate_loader, 3, device, one_batch=True)

# Tiny Residual Network
tiny = TinyClassifier2d()
train_model(tiny, 'tiny', train_loader, validate_loader, 3, device, one_batch=True)
