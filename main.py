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

    tmp_img = alexnet_preprocess(train_dataset[0][0])

    processed_tmp_img = alexnet.avgpool(alexnet.features(tmp_img))
    processed_tmp_img = torch.flatten(processed_tmp_img)




if __name__ == '__main__':
    main()