import sys
import os

sys.path.append(os.path.abspath('../'))

del sys, os

import numpy as np

import torch
from torch.optim import Adam

from torch.utils.data import DataLoader, Subset

from utils import dataset

from tensorboardX import SummaryWriter

import models.vae


def train(model, device, epoch, train_loader, optimiser, writer):

    model.train()

    counts = 0
    running_loss = 0.

    for i, data in enumerate(train_loader):

        n = data.size(0)

        counts += n

        data = data.to(device)

        out, mean, logv = model(data)

        loss = model.loss(data, out, mean, logv)
        loss.backward()
        optimiser.step()

        running_loss += loss.item() * n

        if i % int(len(train_loader) ** .5) or i == len(train_loader) - 1:
            step = epoch * len(train_loader) + i

            # Loss
            writer.add_scalar('train/loss', running_loss / counts, step)
            # print(running_loss / counts)

    writer.add_scalar('train/epoch-loss', running_loss / counts, epoch)


def valid(model, device, epoch, valid_loader, writer):

    model.eval()

    counts = 0
    running_loss = 0.

    with torch.no_grad():

        for i, data in enumerate(valid_loader):

            n = data.size(0)

            counts += n

            data = data.to(device)

            out, mean, logv = model(data)

            loss = model.loss(data, out, mean, logv)

            running_loss += loss.item() * n

            if i % int(len(valid_loader) ** .5) or i == len(valid_loader) - 1:
                step = epoch * len(valid_loader) + i

                # Loss
                writer.add_scalar('valid/loss', running_loss / counts, step)

        writer.add_scalar('valid/epoch-loss', running_loss / counts, epoch)


def main(model, test=False):

    writer = SummaryWriter('logs/vae')

    train_set = dataset.BinarizedMNIST(data_dir='../data/', split='train')
    valid_set = dataset.BinarizedMNIST(data_dir='../data/', split='valid')

    if test:
        train_set = Subset(train_set, indices=np.arange(200))
        valid_set = Subset(valid_set, indices=np.arange(200))

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=10)
    valid_loader = DataLoader(valid_set, batch_size=128, shuffle=True, num_workers=10)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    optimiser = Adam(model.parameters(), lr=3e-4)

    for epoch in range(20):

        train(model, device, epoch, train_loader, optimiser, writer)
        valid(model, device, epoch, valid_loader, writer)

    torch.save(model.state_dict(), 'vae.pth')


if __name__ == '__main__':

    vae = models.vae.VariationalAutoEncoder()

    main(vae, test=False)
