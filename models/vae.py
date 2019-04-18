import torch
from torch import nn


class VariationalAutoEncoder(nn.Module):

    def __init__(self, kappa=1.):
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


        # self.reconstruction_criterion = nn.BCEWithLogitsLoss()
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

        return total_loss, divergence.item(), reconstruction.item(), 0

    # def loss(self, recon_x, x, mu, logvar):
    #     # reconstruction
    #     BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    #
    #     # see Appendix B from VAE paper:
    #     # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    #     # https://arxiv.org/abs/1312.6114
    #     # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #
    #     # KL Divergence
    #     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #
    #     return BCE + KLD

