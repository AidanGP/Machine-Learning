# coding: utf-8

# My fist MNIST Classifier, this does not have GPU acceleration
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets

BS = 10 # Batch size
lr = 0.001 # learning rate
epochs = 3 #number of full iterations through the training set

# data used for training the model
train = datasets.MNIST('../MNIST_DATA', train=True, download=True,
                      transform=transforms.Compose([transforms.ToTensor()]))

# data used for testing the accuracy of the model
test  = datasets.MNIST('../MNIST_DATA', train=False, download=True,
                      transform=transforms.Compose([transforms.ToTensor()]))

# group the data sets into shuffled batches to improve the validity of the model
trainset = torch.utils.data.DataLoader(train, batch_size=BS, shuffle=True)
testset  = torch.utils.data.DataLoader(test,  batch_size=BS, shuffle=True)

# neural network class
class Net(nn.Module): # inherit the torch.nn.Model methods
    def __init__(self):
        super().__init__() # initialise parent class
        self.l1 = nn.Linear(784, 128) # create the linear layers
        self.l2 = nn.Linear(128, 64) # the linear layers manage the weights and biases
        self.l3 = nn.Linear(64, 64) # 4 laters with 784 inputs and 10 outputs.
        self.l4 = nn.Linear(64, 10) # 10 outputs corresponding to 10 digits

    def forward(self, x): # feed forward pass
        x = F.relu(self.l1(x)) # relu is rectified linear function
        x = F.relu(self.l2(x)) # relu is max(0, x)
        x = F.relu(self.l3(x)) # apply relu to the output of every layer
        x = self.l4(x)
        # log softmax returns an array of confidence scores for each digit
        return F.log_softmax(x, dim=1)

def train(model, epochs):
    optimiser = optim.Adam(model.parameters(), lr=lr) # start an optimiser using the Adam algorithm
    for epoch in range(epochs): # iterate through the training set epoch times.
        for data in trainset:   # iterate through the images in the training set
            X, y = data         # set x and y equal to the image and the label respectively
            model.zero_grad() # reset the derivitives to zero
            out = model(X.view(-1, 784)) # pass an image through the network
            loss = F.nll_loss(out, y) # calculate the loss / error / cost
            loss.backward() # calculate the derivitive of the loss function
            optimiser.step() # increment the weights and biases
        return loss

def test(model):
    total, correct = 0, 0
    # we want to iterate through the test set without computing gradients because we arent learning from this
    with torch.no_grad():
        for data in testset: # iterate through the images
            X, y = data # assign x and y to the image and the label respectively
            out = model(X.view(-1, 784)) # pass image to network
            for ind, i in enumerate(out): #check if the returned value from the network matches the label
                correct += (torch.argmax(i) == y[ind]).float() #increment the counters
                total   += 1
    return correct/total
