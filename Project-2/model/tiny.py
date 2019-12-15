from torch import nn
from torchvision import models
from collections import OrderedDict

class TinyClassifier2d(nn.Module):
    def __init__(self, dim=48, num_classes=2, dropout=None):
        super(TinyClassifier2d, self).__init__()

        self.dim = dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(inplace=True)

        # 1, 224, 224

        self.conv1 = nn.Conv2d(1, self.dim, 3, stride=2, padding=1)
        self.leakyrelu1 = nn.LeakyReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(self.dim)
        self.mp1 = nn.MaxPool2d(2, stride=2)

        # 48, 56, 56

        self.conv2 = nn.Conv2d(self.dim, 2 * self.dim, 3, stride=2, padding=1)
        self.leakyrelu2 = nn.LeakyReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(2 * self.dim)
        self.mp2 = nn.MaxPool2d(2, stride=2)

        # 96, 14, 14

        self.conv3 = nn.Conv2d(2 * self.dim, 4 * self.dim, 3, stride=2, padding=1)
        self.leakyrelu3 = nn.LeakyReLU(inplace=True)
        self.bn3 = nn.BatchNorm2d(4 * self.dim)
        self.mp3 = nn.MaxPool2d(2)

        # 192, 3, 3

        self.conv4 = nn.Conv2d(4 * self.dim, 8 * self.dim, 3)
        self.leakyrelu4 = nn.LeakyReLU(inplace=True)
        self.bn4 = nn.BatchNorm2d(8 * self.dim)

        # 384, 1, 1
        self.conv_out = nn.Conv2d(8 * self.dim, self.num_classes, 1)

    def forward(self, input):
        result = self.conv1(input)
        result = self.leakyrelu1(result)
        result = self.bn1(result)
        result = self.mp1(result)

        result = self.conv2(result)
        result = self.leakyrelu2(result)
        result = self.bn2(result)
        result = self.mp2(result)

        result = self.conv3(result)
        result = self.leakyrelu3(result)
        result = self.bn3(result)
        result = self.mp3(result)

        result = self.conv4(result)
        result = self.leakyrelu4(result)
        result = self.bn4(result)

        result = self.conv_out(result)
        result = result.view(-1, self.num_classes)

        return result
