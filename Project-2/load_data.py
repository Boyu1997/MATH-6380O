import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np

class KaggleDataset(Dataset):
    def __init__(self, dataset, transform=None, test=False): # to normalize data, transform = [mean, std] self.left_temps = df['Left_Temps'].values
        self.dataset = dataset
        if test:
            self.imgs = dataset.imgs
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx][0]
        if self.transform:
            data = self.transform(data)
        label = 0 if self.dataset[idx][1] else 1
        if self.test:
            img = self.imgs[idx][0].split("/")[-1].split(".")[0]
            return data, label, img
        else:
            return data, label, 0

def build_balancing_sampler(dataset):
    weights = []
    for data, label, _ in dataset:
        if label:
            weights.append(0.9)
        else:
            weights.append(0.1)
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    return sampler

def load_data(train_validation_split=0.8, batch_size=128, num_workers=4):

    # define dataset
    train_data = datasets.ImageFolder(root='data/train')
    test_data = datasets.ImageFolder(root='data/test')

    # split train dataset into train and validate
    train_size = int(len(train_data) * 0.8)
    validate_size = len(train_data) - train_size
    train_data, validate_data = torch.utils.data.random_split(train_data, [train_size, validate_size])

    # define transform for train, validate, and test
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomAffine(5, translate=(0, 0.1), scale=(0.9, 1.1), shear=(-5, 5),),
        transforms.ToTensor(),
    ])

    validate_test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    # define dataset
    train_dataset = KaggleDataset(train_data, transform=train_transform)
    validate_dataset = KaggleDataset(validate_data, transform=validate_test_transform)
    test_dataset = KaggleDataset(test_data, transform=validate_test_transform, test=True)

    # define sampler for class balancing
    train_sampler = build_balancing_sampler(train_dataset)
    validate_sampler = build_balancing_sampler(validate_dataset)

    # define dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               sampler=train_sampler,
                                               batch_size=batch_size,
                                               num_workers=num_workers)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  sampler=validate_sampler,
                                                  batch_size=batch_size,
                                                  num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              num_workers=num_workers)

    return train_loader, validate_loader, test_loader
