import torch
from torch import nn


class VariationalAutoEncoder(nn.Module):

    def __init__(self, kappa=10.):

        super(VariationalAutoEncoder, self).__init__()

        self.activation = nn.ELU()

        self.encoder_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            self.activation,
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3),
            self.activation,
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 256, kernel_size=5),
            self.activation,
        )
        self.encoder_mean = nn.Linear(256, 100)
        self.encoder_logv = nn.Linear(256, 100)

        self.decoder_input = nn.Linear(100, 256)
        self.decoder_layers = nn.Sequential(
            self.activation,
            nn.Conv2d(256, 64, kernel_size=5, padding=4),
            self.activation,
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=2),
            self.activation,
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(32, 16, kernel_size=3, padding=2),
            self.activation,
            nn.Conv2d(16, 1, kernel_size=3, padding=2),
        )

        self.reconstruction_criterion = nn.CrossEntropyLoss()
        self.kappa = kappa

    def encode(self, x):

        x = self.encoder_layers(x)

        mean = self.encoder_mean(x)
        logv = self.encoder_logv(x)

        return mean, logv

    def sample(self, mean, logv):

        return mean + torch.randn_like(mean) * logv.exp()

    def decode(self, x):

        x = self.decoder_input(x)
        x = self.decoder_layers(x)

        return x

    def forward(self, x):

        mean, logv = self.encode(x)

        z = self.sample(mean, logv)

        out = self.decode(z)

        return out

    def loss(self, x, out, mean, logv):

        # Reconstruction loss
        reconstruction = self.reconstruction_criterion(out, x)

        # KL Divergence
        divergence = 0.5 * (- 1 - logv + mean.pow(2) + logv.exp()).sum(dim=1).mean()

        total_loss = reconstruction + self.kappa * divergence

        return total_loss