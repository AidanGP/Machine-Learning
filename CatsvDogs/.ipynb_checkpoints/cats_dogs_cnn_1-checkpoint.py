import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from tqdm import trange

#Dataset already shuffled
dataset = np.load(os.path.abspath('cats_dogs_data.npy'), allow_pickle=True)

#Segment Dataset into X and y
dataset_X = torch.Tensor([i[0]/255 for i in dataset]).view(-1, 3, 50, 50)
dataset_y = torch.Tensor([j[1] for j in dataset])

#Segment Dataset into training and testing
test_num = 2494
train_X = dataset_X[:-test_num]
train_y = dataset_y[:-test_num]

test_X = dataset_X[-test_num:]
test_y = dataset_y[-test_num:]


class Net(nn.Module):
	def __init__(self):
		super().__init__()

		self.conv1 = nn.Conv2d( in_channels=3 , out_channels=32 , kernel_size=5 )
		self.conv2 = nn.Conv2d( in_channels=32, out_channels=64 , kernel_size=5 )
		self.conv3 = nn.Conv2d( in_channels=64, out_channels=128, kernel_size=5 )

		self.pool  = nn.MaxPool2d( kernel_size=2, stride=2 )

		self.fc1 = nn.Linear( in_features=128*2*2, out_features=128 )
		self.fc2 = nn.Linear( in_features=128, out_features=2 )

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = self.pool(F.relu(self.conv3(x)))

		x = x.view(-1, 128*2*2)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return F.log_softmax(x, dim=1)

model = Net()

BATCH_SIZE = 100
optimiser = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

def train(epochs):
	for epoch in range(epochs):
		for batch in (t := trange(0, len(train_X), BATCH_SIZE)):
			batch_X = train_X[batch:batch+BATCH_SIZE].view(-1, 3, 50, 50)
			batch_y = train_y[batch:batch+BATCH_SIZE]

			model.zero_grad()
			out = model(batch_X)
			loss = loss_fn(out, batch_y)
			loss.backward()
			optimiser.step()
			t.set_description(f'Loss :: {loss}')
	print(f'Epoch {epoch} :: Loss {loss}')

def test():
	with torch.no_grad():
		total, correct = 0, 0
		for img in (t := trange(len(test_X))):
			out = model(test_X[img].view(-1, 3, 50, 50))[0]
			pred = torch.argmax(out)
			real = torch.argmax(test_y[img])
			if pred == real:
				correct += 1
			total += 1
			t.set_description(f'Acc :: {correct/total}')
	return correct/total
train(6)
print(test())
