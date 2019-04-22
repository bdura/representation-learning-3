import torch
from torch import nn

from torch import autograd

import numpy as np


class Discriminator(nn.Module):

    def __init__(self, input_dimension=2, hidden_dimension=10, n_hidden_layers=1, dropout=0, image=False):
        super(Discriminator, self).__init__()

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.image = image
        if image:
            self.input_layer = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3),
                self.activation,
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3),
                self.activation,
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 256, kernel_size=6),
                self.activation
            )
            self.lin_layer = nn.Sequential(nn.Linear(input_dimension, hidden_dimension),
                                           self.activation)
        else:
            self.input_layer = nn.Sequential(
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
        if self.image:
            x = x.squeeze()
            x = self.lin_layer(x)
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
        """

        :param f: The distribution we want to associate one with
        :param g: The distribution we want to associate zero with
        :return:
        """
        objective_f = self(f).log().mean()
        objective_g = (1 - self(g)).log().mean()

        objective = objective_f / 2. + objective_g / 2.

        # Optimisers minimise a loss
        loss = - objective

        return loss

    def distance(self, f, g):
        return np.log(2) - self.loss(f, g)


class Wasserstein(Discriminator):

    def __init__(self, input_dimension=2, hidden_dimension=20, n_hidden_layers=1, dropout=0, kappa=10, image=False):
        super(Wasserstein, self).__init__(
            input_dimension=input_dimension,
            hidden_dimension=hidden_dimension,
            n_hidden_layers=n_hidden_layers,
            dropout=dropout,
            image=image
        )

        self.kappa = kappa

    def loss(self, f, g, penalised=True):
        device = f.device

        objective_f = self(f).mean()
        objective_g = self(g).mean()

        objective = objective_f - objective_g

        if penalised:
            # Uniform distribution U[0, 1]
            a = torch.randn(f.shape[0], ).to(device).unsqueeze(1)

            if len(f.shape) > 2:
                a = a.unsqueeze(2).unsqueeze(2)

            z = a * f + (1 - a) * g
            z.requires_grad_(True)

            gradient = autograd.grad(self(z).sum(), z, create_graph=True)[0]

            norm_gradient = torch.norm(gradient, dim=1)

            penalty = (norm_gradient - 1).pow(2).mean()

            objective = objective - self.kappa * penalty

        loss = - objective

        return loss

    def distance(self, f, g):
        return - self.loss(f, g, penalised=False)
