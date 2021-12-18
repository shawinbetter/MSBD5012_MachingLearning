from __future__ import print_function
from builtins import range

"""
SECTION 1 : Load and setup data for training

the datasets separated in two files from originai datasets:
iris_train.csv = datasets for training purpose, 80% from the original data
iris_test.csv  = datasets for testing purpose, 20% from the original data
"""
import pandas as pd
import numpy as np

#load
datatrain = pd.read_csv('iris_training.csv')

# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"
train_data = np.genfromtxt(IRIS_TRAINING, skip_header=1, 
    dtype=float, delimiter=',') 
test_data = np.genfromtxt(IRIS_TEST, skip_header=1, 
    dtype=float, delimiter=',') 

#split x and y (feature and target)
xtrain = train_data[:,:4]
ytrain = train_data[:,4]

"""
SECTION 2 : Build and Train Model

Multilayer perceptron model, with one hidden layer.
input layer : 4 neuron, represents the feature of Iris
hidden layer : 3 neuron, activation using ReLU
output layer : 3 neuron, represents the class of Iris

optimizer = stochastic gradient descent with no batch-size
loss function = categorical cross entropy
epoch = 50
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1234)

#hyperparameters
hl = 3
lr = 0.01
num_epoch = 2000

#build model
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, hl)
        self.fc2 = nn.Linear(hl, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
net = Net()

#choose optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
#optimizer = torch.optim.Adagrad(net.parameters(), lr=lr)
#optimizer = torch.optim.Adam(net.parameters(), lr=lr)
#optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
#optimizer = torch.optim.RMSprop(net.parameters(), lr=lr)

#train
for epoch in range(num_epoch):
    X = torch.Tensor(xtrain).float()
    Y = torch.Tensor(ytrain).long()

    #feedforward - backprop
    optimizer.zero_grad() #梯度置零
    out = net(X) #前向传播
    loss = criterion(out, Y) #计算loss
    loss.backward() #反向传播求得梯度
    optimizer.step() #优化器更新权重
    acc = 100 * torch.sum(Y==torch.max(out.data, 1)[1]).double() / len(Y)
    if (epoch % 50 == 1):
	    print ('Epoch [%d/%d] Loss: %.4f   Acc: %.4f' 
                   %(epoch+1, num_epoch, loss.item(), acc.item()))


"""
SECTION 3 : Testing model
"""

#split x and y (feature and target)
xtest = test_data[:,:4]
ytest = test_data[:,4]

#get prediction
X = torch.Tensor(xtest).float()
Y = torch.Tensor(ytest).long()
out = net(X)
_, predicted = torch.max(out.data, 1)

#get accuration
print('Accuracy of testing %.4f %%' % (100 * torch.sum(Y==predicted).double() / len(Y)))
