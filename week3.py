import numpy as np
import torch

def train_softmax_regression():
    x_data = torch.Tensor(
            [[10, 30, 20],
             [30, 10, 30],
             [40, 70, 70],
             [80, 0, 0],
             [90, 80, 50],
             [80, 100, 100]])  # [6, 3]

    # x_data = x_data.transpose(1, 0)

    # [6, 3] * [1, 3, 3]

    y_data = torch.tensor(
        [0, 0, 1, 1, 2, 2]
    )

    W = torch.ones((3, 3), requires_grad=True)
    b = torch.zeros(3, requires_grad=True)

    loss_fn = torch.nn.CrossEntropyLoss()


    optimizer = torch.optim.SGD([W, b], lr=0.01)

    nb_epochs = 30000
    for epoch in range(nb_epochs + 1):
        hypothesis = torch.softmax((x_data.unsqueeze(1) * W.unsqueeze(0)).sum(dim=-1) + b.unsqueeze(0), dim=-1)

        loss = loss_fn(hypothesis, y_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print('Epoch {:4d}/{} loss: {:.10f}'.format(
                epoch, nb_epochs, loss.item()
            ))

    print(torch.softmax((x_data.unsqueeze(1) * W.unsqueeze(0)).sum(dim=-1) + b.unsqueeze(0), dim=-1))

train_softmax_regression()