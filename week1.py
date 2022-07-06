import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_data():
    x = [1, 2, 3, 4]
    y = [40, 60, 70, 80]

    plt.scatter(x, y)
    plt.show()


def naive_mse():
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    mean_value = 0

    for i in range(len(x)):
        mean_value += (x[i] - y[i]) ** 2

    print(mean_value/len(x))


def numpy_mse():
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    numpy_x = np.array(x)
    numpy_y = np.array(y)

    print(((numpy_x - numpy_y) ** 2).mean())


def only_w_linear_regression():
    x = np.arange(10)  #
    y = np.arange(10) * 2  # W = 2
    alpha = 0.005  # if alpha is very large or small?
    W = 10

    for iters in range(100):
        mse = ((x * W - y) * x).mean()
        W -= mse * alpha
        if iters % 10 == 0:
            print('Epoch {} W: {:.3f}, Cost: {:.6f}'.format(
                iters, W, mse
            ))


def torch_linear_regression_1():
    x_train = torch.arange(10)  # x 값
    y_train = torch.arange(10) * 3 + 7  # y 값

    # 단순히 숫자를 사용하는 것이 아니라, 값이 변해야 하는 변수라면 requires_grad=True 옵션이 필요
    W = torch.zeros(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    optimizer = torch.optim.SGD([W, b], lr=0.01)  # gradient descent optimizer 설정

    nb_epochs = 1999  # 원하는만큼 경사 하강법을 반복
    for epoch in range(nb_epochs + 1):

        # H(x) 계산
        hypothesis = x_train * W + b

        # cost 계산
        cost = torch.mean((hypothesis - y_train) ** 2)

        # cost로 H(x) 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # 100번마다 로그 출력
        if epoch % 100 == 0:
            print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
                epoch, nb_epochs, W.item(), b.item(), cost.item()
            ))



