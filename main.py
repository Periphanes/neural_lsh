import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torchvision

import argparse
import os

ENCODED_VEC_SIZE_ALEXNET = 9216


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=1004)
parser.add_argument('--cpu', type=int, default=0)

parser.add_argument('--batch-size', type=int, default=16)

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

class simpleHashModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.hash_func = nn.Sequential(
            nn.Linear(ENCODED_VEC_SIZE_ALEXNET, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.hash_func(x)

def main():
    print("Program Started")

    train_dataset = torchvision.datasets.CIFAR10(root="./datasets", train=True, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root="./datasets", train=False, download=True)

    train_selected = []
    test_selected = []

    for data in train_dataset:
        if data[1] == 0 or data[1] == 1:
            train_selected.append(data)
    for data in test_dataset:
        if data[1] == 0 or data[1] == 1:
            test_selected.append(data)

    alexnet = torchvision.models.alexnet(weights = torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
    alexnet_weights = torchvision.models.AlexNet_Weights.IMAGENET1K_V1
    alexnet_preprocess = alexnet_weights.transforms()

    for param in alexnet.parameters():
        param.requires_grad = False

    class AlexDataset(torch.utils.data.Dataset):
        def __init__(self, dataset):
            super().__init__()

            self.data_list = dataset
        
        def __len__(self):
            return len(self.data_list)

        def  __getitem__(self, index):
            img, y = self.data_list[index]

            processed_img = alexnet.avgpool(alexnet.features(alexnet_preprocess(img)))
            processed_img = torch.flatten(processed_img)

            return processed_img, y
    
    alex_train_dataset = AlexDataset(train_selected)
    alex_test_dataset = AlexDataset(test_selected)

    alex_train_loader = torch.utils.data.DataLoader(alex_train_dataset, batch_size=16, shuffle=True, drop_last=True)
    alex_test_loader = torch.utils.data.DataLoader(alex_test_dataset, batch_size=16, shuffle=True, drop_last=True)

    for batch in alex_train_loader:
        print(batch)
        exit(0)


if __name__ == '__main__':
    main()