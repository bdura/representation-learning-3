import torch
from torch import nn
import models.estimators as est
import torch.optim as optim
import base.classify_svhn as data_utils


class GAN(nn.Module):
    def __init__(self, critic=est.Wasserstein()):
        super(GAN, self).__init__()
        # TODO : corriger architecture pour SVHN
        self.critic = critic
        self.generator = nn.Sequential(
            nn.Linear(100, 256),
            self.activation,
            nn.Conv2d(256, 64, kernel_size=5, padding=4),
            self.activation,
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=2),
            self.activation,
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(32, 16, kernel_size=3, padding=2),
            self.activation,
            nn.Conv2d(16, 1, kernel_size=3, padding=2)
        )

    def forward(self, batch_size):
        sample = torch.randn(size=(batch_size, 100))
        generated = self.generator(sample)
        return generated

    def fit(self, num_epochs, device, batch_size):
        self.train()
        discriminator_optim, generator_optim = optim.Adam(params=self.critic.parameters()), optim.Adam(
            params=self.generator.parameters())
        train_loader, valid_loader, test_loader = data_utils.get_data_loader("svhn", batch_size)
        # TODO gérer les loaders valid, test...
        # TODO monitor training (e.g. générer images, metrics)
        for epoch in range(num_epochs):
            for inputs, _ in train_loader:
                inputs = inputs.to(device)

                fake_batch = self(batch_size=batch_size)

                # Compute discriminator predictions
                out_real = self.critic(inputs)
                out_fake = self.critic(fake_batch)

                # Compute losses
                disc_loss = self.critic.loss(out_real, out_fake)
                gen_loss = -disc_loss

                # Backprop discriminator, generator
                disc_loss.backward()
                discriminator_optim.step()

                gen_loss.backward()
                generator_optim.step()

                discriminator_optim.zero_grad()
                generator_optim.zero_grad()


if __name__ == "main":
    gan = GAN()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gan.fit(5, device, 32)
