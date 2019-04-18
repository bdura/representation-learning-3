import torch
from torch import nn
import models.estimators as est


class Encoder(nn.Module):
    def __init__(self, kappa=1.):
        super(Encoder, self).__init__()

        self.activation = nn.ELU()
        # self.sigmoid = nn.Sigmoid()

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

    def forward(self, x):
        x = self.encoder_layers(x)

        x = x.squeeze()
        mean = self.encoder_mean(x)
        logv = self.encoder_logv(x)

        return mean, logv


class Decoder(nn.Module):

    def __init__(self, kappa=1., reconstruction_criterion=nn.BCELoss(), sigmoid=nn.Sigmoid(), activation=nn.ELU()):
        super(Decoder, self).__init__()
        self.reconstruction_criterion = reconstruction_criterion
        self.activation = activation
        self.sigmoid = sigmoid
        self.kappa = kappa

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

    def forward(self, x):
        x = self.decoder_input(x)

        x = x.unsqueeze(2).unsqueeze(2)

        x = self.decoder_layers(x)
        x = self.sigmoid(x)

        return x


class VariationalAutoEncoder(nn.Module):
    def __init__(self, kappa=1., reconstruction_criterion=nn.BCELoss(), sigmoid=nn.Sigmoid(), activation=nn.ELU()):
        super(VariationalAutoEncoder, self).__init__()
        self.reconstruction_criterion = reconstruction_criterion
        self.kappa = kappa

        self.encoder = Encoder()
        self.decoder = Decoder(activation, sigmoid)

    def sample(self, mean, logv):
        sigma = torch.exp(.5 * logv) + .1e-4
        return mean + torch.randn_like(mean) * sigma

    def decode(self, x):
        x = self.decoder_input(x)

        x = x.unsqueeze(2).unsqueeze(2)

        x = self.decoder_layers(x)
        x = self.sigmoid(x)

        return x

    def forward(self, x):
        mean, logv = self.encode(x)
        z = self.sample(mean, logv)
        out = self.decode(z)
        return out, mean, logv

    def loss(self, x, out, mean, logv):
        # Reconstruction loss
        reconstruction = self.reconstruction_criterion(out.view(-1, 784), x.view(-1, 784))

        # KL Divergence
        divergence = 0.5 * (- 1 - logv + mean.pow(2) + logv.exp()).mean()

        total_loss = reconstruction + self.kappa * divergence
        # raise(ValueError('Debugging'))

        return total_loss, divergence.item(), reconstruction.item()

class GAN(nn.Module):
    def __init__(self, kappa=1., reconstruction_criterion=nn.BCELoss(), sigmoid=nn.Sigmoid(), activation=nn.ELU()):
        super(GAN, self).__init__()
        self.reconstruction_criterion = reconstruction_criterion
        self.kappa = kappa

        self.decoder = Decoder(activation, sigmoid)
        self.critic = est.Wasserstein()

    def forward(self, true, sample):
        generated = self.decoder(sample)
        out = self.critic(true, sample)
        return generated, out

    def loss(self, x, out, mean, logv):
        # TODO : create some loss
        reconstruction = self.reconstruction_criterion(out.view(-1, 784), x.view(-1, 784))

        # KL Divergence
        divergence = 0.5 * (- 1 - logv + mean.pow(2) + logv.exp()).mean()

        total_loss = reconstruction + self.kappa * divergence
        # raise(ValueError('Debugging'))

        return total_loss, divergence.item(), reconstruction.item()



class VariationalAutoEncoder_old(nn.Module):

    def __init__(self, kappa=1.):
        super(VariationalAutoEncoder_old, self).__init__()

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

        # self.reconstruction_criterion = nn.BCELoss()
        self.reconstruction_criterion = nn.MSELoss()

        self.sigmoid = nn.Sigmoid()
        self.kappa = kappa

    def encode(self, x):
        x = self.encoder_layers(x)

        x = x.squeeze()

        mean = self.encoder_mean(x)
        logv = self.encoder_logv(x)

        return mean, logv

    def sample(self, mean, logv):
        sigma = torch.exp(.5 * logv) + .1e-4

        # print(mean.size())
        # print('means', mean[0])
        # print('sigmas', sigma[0])
        # print((mean + torch.randn_like(mean) * sigma)[0])
        # print(mean[0,0].item(), logv[0,0].item())
        # return mean + torch.randn_like(mean) / 10

        sigma = torch.exp(.5 * logv)

        return mean + torch.randn_like(mean) * sigma

    def decode(self, x):
        x = self.decoder_input(x)

        x = x.unsqueeze(2).unsqueeze(2)

        x = self.decoder_layers(x)
        x = self.sigmoid(x)

        return x

    def forward(self, x):
        mean, logv = self.encode(x)

        z = self.sample(mean, logv)

        out = self.decode(z)

        return out, mean, logv

    def loss(self, x, out, mean, logv):
        # Reconstruction loss

        reconstruction = self.reconstruction_criterion(out.view(-1, 784), x.view(-1, 784))

        # print()

        # KL Divergence
        divergence = 0.5 * (- 1 - logv + mean.pow(2) + logv.exp()).mean()
        # divergence = 0.5 * (- 1 + mean.pow(2)).mean()
        # print('Divergence', divergence.item())
        # total_loss =  self.kappa * divergence
        # norm = torch.norm(self.encoder_logv.weight)
        # total_loss = reconstruction + self.kappa * divergence + 5 * norm

        total_loss = reconstruction + self.kappa * divergence

        # print('Total', total_loss.item())
        # raise(ValueError('Debugging'))

        return total_loss, divergence.item(), reconstruction.item()
