import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torchvision

import argparse
import os
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=1004)
parser.add_argument('--cpu', type=int, default=0)

parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--val-epoch', type=int, default=5)

parser.add_argument('--hamming-thresh', type=int, default=128)
parser.add_argument('--band-length', type=int, default=16)
parser.add_argument('--hash-sig-length', type=int, default=256)

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if args.cpu or not torch.cuda.is_available():
    device = torch.device('cpu')
else:
    device = torch.device('cuda')

dsh_loss_margin = 2
dsh_loss_alpha = 0.01

def dsh_loss(output1, output2, y):
    loss_sub = (torch.ones_like(y) - y) * torch.linalg.vector_norm(output1 - output2).item() / 2
    loss_add = y * torch.max((dsh_loss_margin - torch.linalg.vector_norm(output1 - output2)), 0)[0] / 2
    loss_relax = dsh_loss_alpha * (torch.linalg.vector_norm(torch.abs(output1) - torch.ones_like(output1), ord = 1)
                                    + torch.linalg.vector_norm(torch.abs(output2) - torch.ones_like(output2), ord = 1))

    return loss_sub + loss_add + loss_relax

print("Device Used : ", device)
args.device = device

class simpleHashModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.hash_func = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(5,5), stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,3), stride=2),
            nn.Conv2d(32,32, kernel_size=(5,5), stride=1, padding=3),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(3,3), stride=2),
            nn.Conv2d(32,64, kernel_size=(5,5), stride=1, padding=3),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(3,3), stride=2),

            nn.Flatten(),

            nn.Linear(1024, 500),
            nn.ReLU(),
            nn.Linear(500, args.hash_sig_length)
        )
    
    def forward(self, x):
        ret = self.hash_func(x)

        return ret
    
def collate_binary(data):
    X_batch = []
    y_batch = []

    X_tmp = None
    y_tmp = None

    for id, data_point in enumerate(data):

        if id % 2 == 0:
            X_tmp = data_point[0]
            y_tmp = data_point[1]
        else:
            if y_tmp == data_point[1]:
                y_batch.append(1)
            else:
                y_batch.append(0)
            X_batch.append(torch.stack((data_point[0], X_tmp)))
    
    X = torch.stack(X_batch).swapaxes(0, 1)
    y = torch.tensor(y_batch)

    return X, y

def main():
    print("Program Started")

    train_dataset = torchvision.datasets.CIFAR10(root="./datasets", train=True, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root="./datasets", train=False, download=True)

    train_selected = []
    test_selected = []

    for data in train_dataset:
        if data[1] == 0 or data[1] == 1:
            train_selected.append((torchvision.transforms.functional.pil_to_tensor(data[0]), data[1]))
    for data in test_dataset:
        if data[1] == 0 or data[1] == 1:
            test_selected.append((torchvision.transforms.functional.pil_to_tensor(data[0]), data[1]))

    train_selected, validation_selected = train_test_split(train_selected, test_size=0.2, random_state=args.seed)

    model = simpleHashModel(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1)

    class AlexDataset(torch.utils.data.Dataset):
        def __init__(self, dataset):
            super().__init__()

            self.data_list = dataset
        
        def __len__(self):
            return len(self.data_list)

        def  __getitem__(self, index):
            img, y = self.data_list[index]

            return img, y
    
    alex_train_dataset = AlexDataset(train_selected)
    alex_val_dataset = AlexDataset(validation_selected)
    alex_test_dataset = AlexDataset(test_selected)

    alex_train_loader = torch.utils.data.DataLoader(alex_train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate_binary)
    alex_val_loader = torch.utils.data.DataLoader(alex_val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate_binary)
    alex_test_loader = torch.utils.data.DataLoader(alex_test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=collate_binary)

    pbar = tqdm(total=args.epochs, initial=0, bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")
    epoch_pbar = tqdm(total=args.epochs, initial=0, bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")

    validation_loss_lst = []

    for epoch_idx in range(args.epochs):
        model.train()
        training_loss = []

        epoch_pbar.reset(len(train_selected) // (args.batch_size // 2))

        for batch_idx, batch in enumerate(alex_train_loader):
            train_x, train_y = batch
            train_x1 = train_x[0,:,:].float().to(device)
            train_x2 = train_x[1,:,:].float().to(device)
            train_y = train_y.to(device)

            optimizer.zero_grad()

            output1 = model(train_x1).squeeze()
            output2 = model(train_x2).squeeze()

            loss = dsh_loss(output1, output2, train_y).sum()
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            scheduler.step()

            training_loss.append(loss.item())

            if epoch_idx < args.val_epoch + 1:
                pbar.set_description("Training Loss : " + str(sum(training_loss) / len(training_loss)))
            else:
                avg_validation_loss = sum(validation_loss) / len(validation_loss)
                pbar.set_description("Training Loss : " + str(sum(training_loss) / len(training_loss)) + " / Val Loss : " + str(avg_validation_loss))

            epoch_pbar.set_description("Training... ")
            epoch_pbar.update(1)
            
            pbar.refresh()
        
        if epoch_idx % args.val_epoch == 0 and epoch_idx != 0:
            model.eval()

            validation_loss = []

            validation_true = np.array([])
            validation_pred = np.array([])

            epoch_pbar.reset(len(validation_selected) // (args.batch_size // 2))

            for batch_idx, batch in enumerate(alex_val_loader):
                val_x, val_y = batch
                val_x1 = val_x[0,:,:].float().to(device)
                val_x2 = val_x[1,:,:].float().to(device)
                val_y = val_y.to(device)

                output1 = model(val_x1).squeeze()
                output2 = model(val_x2).squeeze()

                loss = dsh_loss(output1, output2, val_y).sum()

                validation_loss.append(loss.item())

                pred = (torch.count_nonzero((output1 >= 0).to(torch.long) - (output2 >= 0).to(torch.long), dim = 1) < (args.hash_sig_length - args.hamming_thresh)).to(torch.long).cpu()

                validation_true = np.append(validation_true, val_y.cpu())
                validation_pred = np.append(validation_pred, pred)

                epoch_pbar.set_description("Validating... ")
                epoch_pbar.update(1)
            
            avg_validation_loss = sum(validation_loss) / len(validation_loss)
            validation_loss_lst.append(avg_validation_loss)
            
            pbar.set_description("Training Loss : " + str(sum(training_loss) / len(training_loss)) + " / Val Loss : " + str(avg_validation_loss))
            pbar.refresh()

            print(classification_report(validation_true, validation_pred))
        
        pbar.update(1)

if __name__ == '__main__':
    main()