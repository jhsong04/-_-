import numpy as np
import torch

def sol_1():
    x1_train = torch.tensor([73., 93., 89., 96., 73.])
    x2_train = torch.tensor([80., 88., 91., 98., 66.])
    x3_train = torch.tensor([75., 93., 90., 100., 70.])

    y_train = torch.tensor([152., 185., 180., 196., 142.])

    # requires_grad=True는 변수라는 뜻으로 이해, 변수인 W, b 선언
    W1 = torch.zeros(1, requires_grad=True)
    W2 = torch.zeros(1, requires_grad=True)
    W3 = torch.zeros(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    optimizer = torch.optim.SGD([W1, W2, W3, b], lr=0.00003)  # gradient descent optimizer 설정

    nb_epochs = 10000  # 원하는만큼 경사 하강법을 반복
    for epoch in range(nb_epochs + 1):

        # H(x) 계산
        hypothesis = x1_train * W1 + x2_train * W2 + x3_train * W3 + b

        # cost 계산
        cost = torch.mean((hypothesis - y_train) ** 2)

        # cost로 H(x) 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # 100번마다 로그 출력
        if epoch % 1000 == 0:
            print('Epoch {:4d}/{} W1: {:.3f}, W2: {:.3f}, W3: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
                epoch, nb_epochs, W1.item(), W2.item(), W3.item(), b.item(), cost.item()
            ))
    print(x1_train * W1 + x2_train * W2 + x3_train * W3 + b)


def sol_2():
    x1_train = torch.tensor([[73., 93., 89., 96., 73.], [80., 88., 91., 98., 66.], [75., 93., 90., 100., 70.]])
    # 3 x 5 array

    y_train = torch.tensor([152., 185., 180., 196., 142.])
    # 1 x 5 array


    W1 = torch.zeros((3, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    optimizer = torch.optim.SGD([W1, b], lr=0.00003)  # gradient descent optimizer 설정
    nb_epochs = 10000  # 원하는만큼 경사 하강법을 반복
    for epoch in range(nb_epochs + 1):

        hypothesis = (x1_train * W1).sum(dim=0) + b
        '''
        x1_train * W1 -> [3 x 5] * [1 x 5] = [3 x 5] array
        ex) 3 x 5 array.sum(dim=0) ->
        [[0, 1, 2, 3, 4]
         [0, 2, 4, 6, 8]
         [0, 3, 6, 9, 12]].sum(dim=0) = [0, 6, 12, 18, 24] = [5] array
        '''
        cost = torch.mean((hypothesis - y_train) ** 2)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print('Epoch {:4d}/{} W1: {:.3f}, W2: {:.3f}, W3: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
                epoch, nb_epochs, W1[0].item(), W1[1].item(), W1[2].item(), b.item(), cost.item()
            ))
    print((x1_train * W1).sum(dim=0) + b)

sol_2()