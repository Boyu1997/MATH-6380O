import torch

from load_data import load_data
from model import vgg16
from train import train_model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print ('Using device:', device)


train_loader, validate_loader, test_loader = load_data()

vgg16 = vgg16()

# set one_batch to True for small local cpu testing
train_model(vgg16, 'vgg16', train_loader, validate_loader, 50, device)
