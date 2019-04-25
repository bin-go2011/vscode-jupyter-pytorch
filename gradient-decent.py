#%%
import numpy as np

def function_2(x):
    return x[0]**2 + x[1]**2

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp_val = x[idx]
        # compute f(x+h)
        x[idx] = tmp_val + h
        fxh1 = f(x)
        #compute f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
    return grad

#%%
print(numerical_gradient(function_2, np.array([3.0, 4.0])))

print(numerical_gradient(function_2, np.array([0.0, 2.0])))

print(numerical_gradient(function_2, np.array([3.0, 0.0])))

#%%
def gradient_descent(f, init_x, lr = 0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x

#%%
init_x = np.array([-3.0, 4.0])
gradient_descent(function_2, init_x = init_x, lr=0.1, step_num=100)

#%%
import torch
from torch.autograd import Variable

x = Variable(torch.Tensor([[3.0], [4.0]]), requires_grad=True)
# x = Variable(torch.Tensor([[0.0], [2.0]]), requires_grad=True)
# x = Variable(torch.Tensor([[3.0], [0.0]]), requires_grad=True)
x = Variable(torch.Tensor([3.0, 4.0]), requires_grad=True)

y = x.pow(2).sum()
y.backward()
print(x.grad)


#%%
