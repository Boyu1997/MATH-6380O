from torch import nn
from torchvision import models
from collections import OrderedDict

def vgg16():
    vgg16 = models.vgg16(pretrained=True)

    for param in vgg16.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 2048)),
                                            ('relu1', nn.ReLU()),
                                            ('fc2', nn.Linear(2048,2048)),
                                            ('relu2', nn.ReLU()),
                                            ('fc3', nn.Linear(2048,2)),
                                            ('output', nn.LogSoftmax(dim=1))
                                           ]))
    vgg16.classifier = classifier

    return vgg16
