#%%
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0

def forward(x):
    return x*w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred-y)*(y_pred-y)

def gradient(x, y):
    return 2*x*(x*w-y)

print("my prediction before training", 4, forward(4))

for epoch in range(100):
    for x_val, y_val in zip(x_data, y_data):
        grad = gradient(x_val, y_val)
        w = w - 0.01*grad
        print("\tgrad: ", x_val, y_val, round(grad, 2))
        l = loss(x_val, y_val)
    print("progress:", epoch, "w=", round(w, 2), "loss=", round(l, 2))
print("my new prediction after training", forward(4))

#%%
import torch
from torch.autograd import Variable

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = Variable(torch.Tensor([1.0]), requires_grad=True)

def forward(x):
    return x*w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred-y)*(y_pred-y)

print("my prediction before training:", 4, forward(4).data[0])

for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        l = loss(x_val, y_val)
        l.backward()
        print("\tgrad: ", x_val, y_val, w.grad.data[0])
        w.data = w.data - 0.01*w.grad.data;

        w.grad.data.zero_()
    print("my progress:", epoch, l.data[0])
print("my prediction after training", forward(4).data[0])

#%%
import torch
import torch.nn as nn
from torch.autograd import Variable

x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))

class LinearModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        out = self.linear(x)
        return out

model = LinearModel(1,1)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    y_pred = model(x_data)

    loss = criterion(y_pred, y_data)
    print(epoch, loss.data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

output = Variable(torch.Tensor([4.0]))
y_pred = model(output)
print("my prediction after training", 4, model(output).data[0])
