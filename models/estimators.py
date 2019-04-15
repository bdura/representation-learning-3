import torch
from torch import nn

from torch import autograd

import numpy as np


class Discriminator(nn.Module):

    def __init__(self, input_dimension=2, hidden_dimension=10, n_hidden_layers=3, dropout=.1):

        super(Discriminator, self).__init__()

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.input_layer = nn.Sequential(
            self.dropout,
            nn.Linear(input_dimension, hidden_dimension),
            self.activation
        )

        self.hidden_layers = nn.Sequential(*[
            nn.Sequential(
                self.dropout,
                nn.Linear(hidden_dimension, hidden_dimension),
                self.activation
            )
            for _ in range(n_hidden_layers)
        ])

        self.output_layer = nn.Sequential(
            self.dropout,
            nn.Linear(hidden_dimension, 1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)

        return x


class JensenShannon(Discriminator):

    def __init__(self, input_dimension=2, hidden_dimension=10, n_hidden_layers=3, dropout=0):

        super(JensenShannon, self).__init__(
            input_dimension=input_dimension,
            hidden_dimension=hidden_dimension,
            n_hidden_layers=n_hidden_layers,
            dropout=dropout
        )

    def forward(self, x):

        x = super().forward(x)
        x = self.sigmoid(x)

        return x

    def loss(self, f, g):

        assert f.size() == g.size(), 'The number of samples for P_f and P_g must be equal.'

        objective_f = self(f).log().mean()
        objective_g = (1 - self(g)).log().mean()

        objective = objective_f / 2. + objective_g / 2.

        # Optimisers minimise a loss
        loss = - objective

        return loss

    def distance(self, f, g):

        return np.log(2) - self.loss(f, g)


class Wasserstein(Discriminator):

    def __init__(self, input_dimension=2, hidden_dimension=10, n_hidden_layers=3, dropout=0, kappa=10):

        super(Wasserstein, self).__init__(
            input_dimension=input_dimension,
            hidden_dimension=hidden_dimension,
            n_hidden_layers=n_hidden_layers,
            dropout=dropout
        )

        self.kappa = kappa

    def loss(self, f, g, penalised=True):

        assert f.size() == g.size(), 'The number of samples for P_f and P_g must be equal.'

        n = f.size(0)
        device = f.device

        # Uniform distribution U[0, 1]
        a = torch.rand((n, 1)).to(device)

        objective_f = self(f).mean()
        objective_g = self(g).mean()

        z = a * f + (1 - a) * g
        z.requires_grad_(True)

        autograd.backward(self(z), z, create_graph=True)
        gradient = z.grad

        norm_gradient = torch.norm(gradient, p=2, dim=1)

        penalty = (norm_gradient - 1).pow(2).mean()

        if penalised:
            objective = objective_f - objective_g - self.kappa * penalty

        else:
            objective = objective_f - objective_g

        loss = - objective

        return loss

    def distance(self, f, g):
        return - self.loss(f, g, penalised=False)


if __name__ == '__main__':

    w = Wasserstein()

    f = 1. + torch.randn((20, 2))
    g = 2 * torch.randn((20, 2))

    loss = w.loss(f, g)

    loss.backward()

    print(w.input_layer)
