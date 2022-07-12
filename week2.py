import numpy as np
import matplotlib.pyplot as plt
import torch

def mat_cal_1():
    x = np.arange(1, 16)  # 1 ~ 15
    x = x.reshape(3, 5)

    y = np.arange(3)  # 0 ~ 2
    new_y = y.reshape(3, 1)

    print(x * new_y)
    # print(x * y) if you execute this line, error will be occurred

def week1_problem_repeat():
    x = np.arange(1, 16).reshape(3, 5)

    weight = np.arange(3).reshape(3, 1)

    bias = np.ones(1)  # just a value

    result = (x * weight).sum(axis=0) + bias  # numpy -> axis, torch -> dim
    print("Before sum:", x * weight)
    print("After sum:", (x * weight).sum(axis=0))
    print("Result:", result)

def train_logistic_regression():
    x_data = torch.Tensor(
            [[1, 2],
             [2, 3],
             [3, 1],
             [4, 3],
             [5, 3],
             [6, 2]])

    # x_data = x_data.transpose(1, 0)  # reshaping data

    y_data = torch.Tensor(
        [[0],
         [0],
         [0],
         [1],
         [1],
         [1]]
    )

    W = torch.zeros((2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    optimizer = torch.optim.SGD([W, b], lr=0.1)
    nb_epochs = 10000
    for epoch in range(nb_epochs + 1):
        hypothesis = torch.sigmoid(x_data.matmul(W) + b)  # if not reshaping?


        cost = torch.mean(-(y_data * torch.log(hypothesis) + (1 - y_data) * torch.log(1 - hypothesis)))

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print('Epoch {:4d}/{} W1: {:.3f}, W2: {:.3f}, b: {:.3f} Cost: {:.10f}'.format(
                epoch, nb_epochs, W[0].item(), W[1].item(), b.item(), cost.item()
            ))

    print(hypothesis)


train_logistic_regression()


