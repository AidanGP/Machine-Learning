#!/usr/bin/env python
# coding: utf-8

# MNIST Classifier with a CNN, GPU acceleration and data augmentation to prevent overfitting.
# Used the nn.Sequential method for the first time (it seems good)


import torch
import torchvision
from torchvision import transforms, datasets
from tqdm import trange, tqdm
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1), shear=10, scale=(0.9, 1.1)),
        transforms.ToTensor(),
        #transforms.CenterCrop(22)
        transforms.Normalize((0.1307,), (0.3081,))
    ]
)


train = datasets.MNIST('../MNIST_DATA', train=True, download=True,
                      transform=transforms.Compose([transforms.ToTensor()]))

test  = datasets.MNIST('../MNIST_DATA', train=False, download=True,
                      transform=transforms.Compose([transforms.ToTensor()]))
trainset = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True)
testset  = torch.utils.data.DataLoader(test,  batch_size=100, shuffle=True)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear(64*3*3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(84, 10)
        )
    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 64*3*3)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)



def train(model, epochs):
    #losses, accuracies = [], []
    optimiser = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        for X, y in tqdm(trainset):
            X, y = X.cuda(), y.cuda()
            model.zero_grad()
            out = model(X)
            loss = F.nll_loss(out, y)
            #losses.append(loss)
            #preds = [torch.argmax(i) for i in out]
            #acc = np.array([int(a)==int(b) for a,b in zip(preds,y)]).mean()
            #accuracies.append(acc)
            loss.backward()
            optimiser.step()
    return losses, accuracies

def test(model):
    accs = []
    for X, y in tqdm(testset):
        X, y = X.cuda(), y.cuda()
        out = model(X)
        preds = [torch.argmax(i) for i in out]
        acc = np.array([int(a)==int(b) for a,b in zip(preds,y)]).mean()
        accs.append(acc)
    return accs