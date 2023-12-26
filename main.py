import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torchvision

import PIL


ENCODED_VEC_SIZE_ALEXNET = 9216


def main():
    print("Program Started")

    train_dataset = torchvision.datasets.CIFAR10(root="./datasets", train=True, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root="./datasets", train=False, download=True)

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
    
    alex_train_dataset = AlexDataset(train_dataset)
    alex_test_dataset = AlexDataset(test_dataset)

    alex_train_loader = torch.utils.data.DataLoader(alex_train_dataset, batch_size=16, shuffle=True, drop_last=True)
    alex_test_loader = torch.utils.data.DataLoader(alex_test_dataset, batch_size=16, shuffle=True, drop_last=True)

    

if __name__ == '__main__':
    main()