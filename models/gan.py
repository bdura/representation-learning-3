import sys
import os

sys.path.append(os.path.abspath('../'))

del sys, os

import torch
from torch import nn
import models.estimators as est
import torch.optim as optim
import base.classify_svhn as data_utils
from tqdm import tqdm
import warnings
from tensorboardX import SummaryWriter
from torch.nn.functional import sigmoid


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
        self.criterion = nn.BCELoss()

    def forward(self, batch_size):
        sample = torch.randn(size=(batch_size, 100))
        sample = self.linear_layer(sample).unsqueeze(2).unsqueeze(2)
        generated = self.generator(sample)
        return generated

    def fit(self, train_loader, num_epochs, device, writer):
        self.train()
        discriminator_optim, generator_optim = optim.Adam(params=self.critic.parameters()), optim.Adam(
            params=[{'params': self.generator.parameters()}, {'params': self.linear_layer.parameters()}])
        # TODO monitor training (e.g. générer images, metrics)
        for epoch in range(num_epochs):
            print("Starting epoch {}...".format(epoch))
            for i, (inputs, _) in tqdm(enumerate(train_loader)):
                inputs = inputs.to(device)

                fake_batch = self(batch_size=inputs.shape[0])

                # Compute losses
                disc_loss = self.critic.loss(inputs, fake_batch)
                gen_loss = -disc_loss

                # Backprop discriminator, generator
                disc_loss.backward(retain_graph=True)
                discriminator_optim.step()

                gen_loss.backward()
                generator_optim.step()

                discriminator_optim.zero_grad()
                generator_optim.zero_grad()

                if i % 20 == 0:
                    out_real = sigmoid(self.critic(inputs))
                    out_fake = sigmoid(self.critic(fake_batch))
                    fake_labels = torch.zeros_like(out_fake)
                    real_labels = torch.ones_like(out_real)
                    clf_loss_fake = self.criterion(out_fake, fake_labels)
                    clf_loss_real = self.criterion(out_real, real_labels)
                    clf_loss = (clf_loss_fake + clf_loss_real) / 2
                    print("cross-entropy loss: {:.4f}, WGAN penalty: {:.4f} after {} batches".
                          format(clf_loss, disc_loss, i))

                    # Log metrics into tensorboard
                    step = epoch * len(train_loader) + i
                    writer.add_scalar('train/bce-loss', clf_loss, step)
                    writer.add_scalar('train/WGAN-penalty', disc_loss, step)

                    # Sample, log and save images
                    image_sample = self(batch_size=1).squeeze(0)
                    writer.add_image('Generator samples', image_sample, step)

            print("Discriminator WGAN loss after {} epochs: {:.4f}".format(epoch, disc_loss))

        torch.save(self.state_dict(), '../learning/gan.pth')


if __name__ == '__main__':
    writer = SummaryWriter('../learning/logs/gan')
    warnings.filterwarnings("ignore", category=UserWarning)
    gan = GAN()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    train_loader = data_utils.get_data_loader("svhn", batch_size)
    gan.fit(train_loader, 1, device, writer)
