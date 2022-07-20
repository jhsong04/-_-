import numpy as np
import torch

def train_softmax_regression():
    x_data = torch.Tensor(
        [[1, 2, 1, 1],
         [2, 1, 3, 2],
         [3, 1, 3, 4],
         [4, 1, 5, 5],
         [1, 5, 5, 5],
         [1, 2, 5, 6],
         [1, 6, 6, 6],
         [1, 7, 7, 7]])  # [6, 3]

    # x_data = x_data.transpose(1, 0)

    # [6, 3] * [1, 3, 3]

    y_data = torch.tensor(
        [2, 2, 2, 1, 1, 1, 0, 0]
    )

    W = torch.ones((3, 4), requires_grad=True)
    b = torch.zeros(3, requires_grad=True)

    loss_fn = torch.nn.CrossEntropyLoss()


    optimizer = torch.optim.SGD([W, b], lr=0.1)

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