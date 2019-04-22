import torch
from torch import nn
import models.estimators as est
import torch.optim as optim
import base.classify_svhn as data_utils


class GAN(nn.Module):
    def __init__(self,
                 critic=est.Wasserstein(input_dimension=256, hidden_dimension=100, n_hidden_layers=0, image=True)):
        super(GAN, self).__init__()
        self.critic = critic
        self.linear_layer = nn.Sequential(nn.Linear(100, 256),
                                          nn.ELU())

        self.generator = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=4, padding=4),
            nn.ELU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=2),
            nn.ELU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(32, 16, kernel_size=3, padding=2),
            nn.ELU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=2)
        )

    def forward(self, batch_size):
        sample = torch.randn(size=(batch_size, 100))
        sample = self.linear_layer(sample).unsqueeze(2).unsqueeze(2)
        generated = self.generator(sample)
        return generated

    def fit(self, num_epochs, device, batch_size):
        self.train()
        discriminator_optim, generator_optim = optim.Adam(params=self.critic.parameters()), optim.Adam(
            params=[{'params': self.generator.parameters()}, {'params': self.linear_layer.parameters()}])
        train_loader = data_utils.get_data_loader("svhn", batch_size)
        # TODO gérer les loaders valid, test...
        # TODO monitor training (e.g. générer images, metrics)
        for epoch in range(num_epochs):
            for inputs, _ in train_loader:
                inputs = inputs.to(device)

                fake_batch = self(batch_size=batch_size)

                # Compute losses
                disc_loss = self.critic.loss(inputs, fake_batch, device)
                gen_loss = -disc_loss

                # Backprop discriminator, generator
                disc_loss.backward(retain_graph=True)
                discriminator_optim.step()

                gen_loss.backward()
                generator_optim.step()

                discriminator_optim.zero_grad()
                generator_optim.zero_grad()

            print("Discriminator loss after {} epochs: {:.4f}".format(epoch, disc_loss))


if __name__ == '__main__':
    gan = GAN()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gan.fit(5, device, 64)
