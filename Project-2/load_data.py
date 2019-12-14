import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np

class KaggleDataset(Dataset):
    def __init__(self, dataset): # to normalize data, transform = [mean, std] self.left_temps = df['Left_Temps'].values
        self.dataset = dataset
        self.imgs = dataset.imgs

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx][0]
        label = 0 if self.dataset[idx][1] else 1
        img = self.imgs[idx][0].split("/")[-1].split(".")[0]
        return data, label, img

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

def load_data(train_validation_split=0.8, batch_size=32, num_workers=4):

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # define dataset
    train_data = datasets.ImageFolder(root='data/train',
                                         transform=transform)
    test_data = datasets.ImageFolder(root='data/test',
                                        transform=transform)

    # define dataset
    train_dataset = KaggleDataset(train_data)
    test_dataset = KaggleDataset(test_data)

    # split train dataset into train and validate
    train_size = int(len(train_dataset) * 0.8)
    validate_size = len(train_dataset) - train_size
    train_dataset, validate_dataset = torch.utils.data.random_split(train_dataset, [train_size, validate_size])

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
