#!/usr/bin/env python
# coding: utf-8

# MNIST Classifier with GPU acceleration.

# In[ ]:


import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
np.set_printoptions(suppress=True)

batch_size = 128

train = datasets.MNIST('', train=True, download=True,
                      transform=transforms.Compose([transforms.ToTensor()]))

test  = datasets.MNIST('', train=False, download=True,
                      transform=transforms.Compose([transforms.ToTensor()]))
trainset = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
testset  = torch.utils.data.DataLoader(test,  batch_size=batch_size, shuffle=True)


# In[ ]:


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


# In[ ]:


def train(model, epochs):
    accuracies, losses = [], []
    optimiser = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        for data in tqdm(trainset):
            X, y = data
            X, y = X.cuda(), y.cuda()
            model.zero_grad()
            output = model(X.view(-1, 784))
            loss = F.nll_loss(output, y)
            losses.append(loss)
            preds = [torch.argmax(out) for out in output]
            acc = np.array([(int(a) == int(b)) for a, b in zip(preds, y)]).mean()
            accuracies.append(acc)
            loss.backward()
            optimiser.step()
    return losses, accuracies


# In[ ]:


model = Net().cuda()
losses, accuracies = train(model, 50)


# In[ ]:


plt.ylim(-0.1, 1.1)
plt.plot(losses)
plt.plot(accuracies)


# In[ ]:


print(f'Loss :: {losses[-1]} Accuracy :: {accuracies[-1]}')


# In[ ]:


def test(model):
    losses, accuracies = [], []
    for data in tqdm(testset):
        X, y = data
        X, y = X.cuda(), y.cuda()
        out = model(X.view(-1, 784))
        loss = F.nll_loss(out, y)
        preds = [torch.argmax(i) for i in out]
        acc = np.array([ (int(a) == int(b)) for a, b in zip(preds, y) ]).mean()
        losses.append(loss)
        accuracies.append(acc)
    return losses, accuracies


# In[ ]:


loss, acc = test(model)


# In[ ]:


plt.plot(loss)
plt.plot(acc)


# In[ ]:


print(f'Loss :: {sum(loss)/len(loss)} Accuracy :: {sum(acc)/len(acc)}')

