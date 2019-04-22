import sys
import os

sys.path.append(os.path.abspath('../'))

del sys, os

import torch
from torch.optim import Adam

from torch.utils.data import DataLoader
import base.classify_svhn as data_utils
from utils import dataset

from tensorboardX import SummaryWriter

import models.vae
import time


def train(model, device, epoch, train_loader, optimiser, writer):
    model.train()

    counts = 0
    running_loss = 0.
    start_time = time.time()

    for i, (data, _) in enumerate(train_loader):
        data = data.to(device)

        n = data.size(0)

        counts += n

        out, mean, logv = model(data)

        loss, div, rec = model.loss(data, out, mean, logv)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        running_loss += loss.item() * n

        if i % 20 == 0:
            time_elapsed = time.time() - start_time
            print("loss = {}, div = {}, rec = {}, time = {}, batch = {}".format(loss.item(), div, rec,
                                                                                time_elapsed, i // 20))
    writer.add_scalar('train/epoch-loss', running_loss / counts, epoch)
    print(running_loss / counts)


def valid(model, device, epoch, valid_loader, writer):
    model.eval()

    counts = 0
    running_loss = 0.

    with torch.no_grad():

        for i, (data, _) in enumerate(valid_loader):
            data = data.to(device)

            n = data.size(0)

            counts += n

            data = data.to(device)

            out, mean, logv = model(data)

            loss, div, rec = model.loss(data, out, mean, logv)

            running_loss += loss.item() * n

            if i % int(len(valid_loader) ** .5) or i == len(valid_loader) - 1:
                step = epoch * len(valid_loader) + i

                # Loss
                writer.add_scalar('valid/loss', running_loss / counts, step)

                writer.add_image('generator sample', model.sample_image(device), step)

        writer.add_scalar('valid/epoch-loss', running_loss / counts, epoch)
        print('loss: ', running_loss / counts)


def main(model, num_epochs, train_loader, valid_loader, name='vae'):
    writer = SummaryWriter('logs/' + name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    optimiser = Adam(model.parameters(), lr=3e-4)

    for epoch in range(num_epochs):
        train(model, device, epoch, train_loader, optimiser, writer)
        valid(model, device, epoch, valid_loader, writer)

    torch.save(model.state_dict(), name + '.pth')


def main_eval(model, test=False):
    writer = SummaryWriter('logs/vae_test/')
    valid_set = dataset.BinarizedMNIST(data_dir='../data/', split='valid', test=test)
    test_set = dataset.BinarizedMNIST(data_dir='../data/', split='test', test=test)

    valid_loader = DataLoader(valid_set, batch_size=128, shuffle=True, num_workers=10)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=True, num_workers=10)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    valid(model, device, 1, valid_loader, writer)
    valid(model, device, 1, test_loader, writer)


if __name__ == '__main__':
    svhn = True
    test = False
    batch_size = 64
    eval = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    map = ('cpu' if device == torch.device('cpu') else None)
    model = models.vae.VariationalAutoEncoder(svhn=svhn)

    if svhn:
        train_loader, valid_loader, test_loader = data_utils.get_data_loader("../models/svhn", batch_size)
        name = 'vae_svhn'
    else:
        train_set = dataset.BinarizedMNIST(data_dir='../data/', split='train', test=test)
        valid_set = dataset.BinarizedMNIST(data_dir='../data/', split='valid', test=test)
        train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=10)
        valid_loader = DataLoader(valid_set, batch_size=128, shuffle=True, num_workers=10)
    if eval:
        model.load_state_dict(torch.load('vae.pth', map_location=map))
        main_eval(model, test=test)
    else:
        main(model, 40, train_loader, valid_loader, name=name)
