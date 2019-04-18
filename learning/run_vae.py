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
import time


def train(model, device, epoch, train_loader, optimiser, writer):

    model.train()

    counts = 0
    running_loss = 0.
    start_time = time.time()

    for i, data in enumerate(train_loader):
        data = data.to(device)

        n = data.size(0)

        counts += n

        data = data.to(device)

        out, mean, logv = model(data)

        loss, div, rec, norm = model.loss(data, out, mean, logv)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        running_loss += loss.item() * n

        if i % 20 == 0:
            time_elapsed = time.time() - start_time
            print(f"loss = {loss.item()}, div = {div}, rec = {rec}, norm = {norm}, time 20 batch = {time_elapsed}")

        """
        if i % int(len(train_loader) ** .5) or i == len(train_loader) - 1:
            step = epoch * len(train_loader) + i

            # Loss
            writer.add_scalar('train/loss', running_loss / counts, step)
            # print(running_loss / counts)
        """
    writer.add_scalar('train/epoch-loss', running_loss / counts, epoch)
    print(running_loss / counts)


def valid(model, device, epoch, valid_loader, writer):

    model.eval()

    counts = 0
    running_loss = 0.

    with torch.no_grad():

        for i, data in enumerate(valid_loader):
            data = data.to(device)

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

    train_set = dataset.BinarizedMNIST(data_dir='../data/', split='train', test=test)
    valid_set = dataset.BinarizedMNIST(data_dir='../data/', split='valid', test=test)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=10)
    valid_loader = DataLoader(valid_set, batch_size=128, shuffle=True, num_workers=10)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    optimiser = Adam(model.parameters(), lr=1e-5)

    for epoch in range(200):

        train(model, device, epoch, train_loader, optimiser, writer)
        valid(model, device, epoch, valid_loader, writer)

    torch.save(model.state_dict(), 'vae.pth')


def ll_importance_sample_batch(batch, model, samples, device):
    """
    :param batch: tensor whose likelihood is to be evaluated shape : (size_minibatch, D)
    :param model: a VAE
    :param samples : number of importance samples per example
    :param device : the usual
    :return: the ll
    """
    model.to(device)
    batch.to(device)
    model.eval()
    with torch.no_grad:
        # Get the means and sigma
        out, mean, logv = model(batch)
        sigma = torch.exp(.5 * logv) + .1e-8
        m = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        samples_value_normal = torch.randn((samples, batch.size()[0], mean.size()[1]))

        # Get the importance weights
        proba_q = m.cdf(samples_value_normal)
        aggregated_q = proba_q.log().sum(dim=2)
        aggregated_q.size()

        # Get the hidden values and their prior proba
        value_z = mean + samples_value_normal * sigma
        proba_prior_z = m.cdf(value_z)
        aggregated_z = proba_prior_z.log().sum(dim=2)

        # Forward pass these hidden values and get the probability of the observed values using the model
        cond_proba = list()
        for i in range(value_z.size()[1]):

            # Do this for each original points, for all samples
            estimated_conditional_point = model.decode(value_z[:, i, :]).view(-1, 784)
            # We get a (samples, 784) tensor
            # print(estimated_conditional_point.size())
            # print(estimated_conditional_point[0])

            # Compute the conditional likelihood of this image
            x = batch[i].view(-1, 784)
            proba_conditional = x * estimated_conditional_point + (1 - x) * (1 - estimated_conditional_point)
            aggregated_cond = proba_conditional.log().sum(dim=1)

            # print(proba_conditional.size())
            # print(aggregated_cond.size())
            cond_proba.append(aggregated_cond)

        cond_proba_tensor = torch.stack(cond_proba).transpose(0, 1)
        proba_points = (aggregated_z + cond_proba_tensor - aggregated_q).exp()

        # Average over the samples
        proba_points.mean(dim=0)

        # Return result
    return proba_points


if __name__ == '__main__':
    pass
    # model = models.vae.VariationalAutoEncoder()
    # main(model, test=False)

    '''
    
    import models.vae
    model = models.vae.VariationalAutoEncoder()
    model = model.to(device)

    for epoch in range(20):
        train(model, device, epoch, train_loader, optimiser, writer)
        valid(model, device, epoch, valid_loader, writer)
        
    '''

