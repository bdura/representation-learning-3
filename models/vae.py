import torch
from torch import nn
from torchvision.transforms import Compose, Normalize


class VariationalAutoEncoder(nn.Module):

    def __init__(self, kappa=1., svhn=False):
        super(VariationalAutoEncoder, self).__init__()
        self.svhn = svhn

        if svhn:
            self.activation = nn.ReLU()
            self.encoder_layers = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3),
                self.activation,
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3),
                self.activation,
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 256, kernel_size=6),
                self.activation
            )

            self.decoder_layers = nn.Sequential(
                nn.Conv2d(256, 64, kernel_size=4, padding=4),
                self.activation,
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(64, 32, kernel_size=3, padding=2),
                self.activation,
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(32, 16, kernel_size=3, padding=2),
                self.activation,
                nn.Conv2d(16, 3, kernel_size=3, padding=2)
            )
            self.reconstruction_criterion = nn.MSELoss(reduction="sum")
        else:
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
            self.reconstruction_criterion = nn.BCELoss(reduction="sum")

        self.encoder_mean = nn.Linear(256, 100)
        self.encoder_logv = nn.Linear(256, 100)

        self.decoder_input = nn.Linear(100, 256)

        self.sigmoid = nn.Sigmoid()
        self.kappa = kappa

    def encode(self, x):
        x = self.encoder_layers(x)

        x = x.squeeze()

        mean = self.encoder_mean(x)
        logv = self.encoder_logv(x)

        return mean, logv

    def perturb(self, idx, std):
        vec = torch.randn(size=(1, 100))
        zeros = torch.zeros_like(vec)
        epsilon = torch.randn(1) * std
        zeros[0, idx] = epsilon
        perturbed_vec = vec + zeros
        sample = self.decode(vec)
        sample = (sample / 2.0) + 0.5
        perturbed_sample = self.decode(perturbed_vec)
        perturbed_sample = (perturbed_sample / 2.0) + 0.5
        return sample.squeeze(0), perturbed_sample.squeeze(0)

    def sample_image(self, device):
        code = torch.randn((1, 100)).to(device)
        sample = self.decode(code)
        if self.svhn:
            img = (sample / 2.0) + 0.5
            img = img.squeeze(0)
        else:
            img = sample
        return img

    def sample(self, mean, logv):
        sigma = torch.exp(.5 * logv)

        return mean + torch.randn_like(mean) * sigma

    def decode(self, x):
        x = self.decoder_input(x)

        x = x.unsqueeze(2).unsqueeze(2)

        x = self.decoder_layers(x)
        if not self.svhn:
            x = self.sigmoid(x)

        return x

    def forward(self, x):
        mean, logv = self.encode(x)

        z = self.sample(mean, logv)

        out = self.decode(z)

        return out, mean, logv

    def loss(self, x, out, mu, logv):
        # Reconstruction loss

        if self.svhn:
            dim1 = 32 * 32
        else:
            dim1 = 28 * 28
        reconstruction = self.reconstruction_criterion(out.view(-1, dim1), x.view(-1, dim1)) / mu.shape[0]

        # KL Divergence
        divergence = 0.5 * (- 1 - logv + mu.pow(2) + logv.exp()).sum(1).mean()

        total_loss = reconstruction + self.kappa * divergence

        return total_loss, divergence.item(), reconstruction.item()
