#!/usr/bin/env python
# coding: utf-8

# MNIST Classifier using a CNN and GPU acceleration.
import torch
import torchvision
from torchvision import transforms, datasets
from tqdm import trange, tqdm
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

batch_size = 128

train = datasets.MNIST('../MNIST_DATA', train=True, download=True,
                      transform=transforms.Compose([transforms.ToTensor()]))

test  = datasets.MNIST('../MNIST_DATA', train=False, download=True,
                      transform=transforms.Compose([transforms.ToTensor()]))
trainset = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
testset  = torch.utils.data.DataLoader(test,  batch_size=batch_size, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)

        self.fc1 = nn.Linear(64*4*4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64*4*4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(model, epochs):
    losses, accuracies = [], []
    optimiser = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        for data in tqdm(trainset):
            X, y = data
            X, y = X.cuda(), y.cuda()
            model.zero_grad()
            output = model(X)
            loss = F.nll_loss(output, y)
            losses.append(loss)
            predictions = [torch.argmax(out) for out in output]
            acc = np.array([int(a) == int(b) for a, b in zip(predictions, y)]).mean()
            accuracies.append(acc)
            loss.backward()
            optimiser.step()
    return losses, accuracies

def test(model):
    losses, accuracies = [], []
    for data in tqdm(testset):
        X, y = data
        X, y = X.cuda(), y.cuda()
        out = model(X)
        loss = F.nll_loss(out, y)
        preds = [torch.argmax(i) for i in out]
        acc = np.array([ (int(a) == int(b)) for a, b in zip(preds, y) ]).mean()
        losses.append(loss)
        accuracies.append(acc)
    return losses, accuracies