#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import matplotlib.pyplot as plt
from torch.autograd.variable import Variable

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2 * torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)

# torch can only train on Variable, so convert them to Variable
x, y = Variable(x), Variable(y)


class Net(torch.nn.Module):
    def __init__(self, input, hidden, output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(input, hidden)
        self.active = torch.nn.ReLU()
        self.out = torch.nn.Linear(hidden, output)

    def forward(self, input):
        x = self.hidden(input)
        x = self.active(x)
        x = self.out(x)
        return x


net1 = Net(1, 10, 1)


def Net2(input, hidden, output):
    return torch.nn.Sequential(
        torch.nn.Linear(input, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, output)
    )


net2 = Net2(1, 10, 1)


print(net1)
print(net2)

optimizer1 = torch.optim.SGD(net1.parameters(), lr=0.5)
optimizer2 = torch.optim.SGD(net2.parameters(), lr=0.5)
lossFunc = torch.nn.MSELoss()
prediction = Variable(torch.FloatTensor(y.data.size()).zero_())
for i in range(100):
    prediction = net1(x)
    loss = lossFunc(prediction, y)
    print(loss)
    optimizer1.zero_grad()
    loss.backward()
    optimizer1.step()
plt.scatter(x.data.numpy(), y.data.numpy())
plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
plt.show()
