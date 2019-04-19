import sys
import os

sys.path.append(os.path.abspath('../'))

del sys, os

import models.vae
import torch
from torch.utils.data import DataLoader
import utils.dataset as dataset


def ll_importance_sample_batch(batch, model, samples):
    """
    :param batch: tensor whose likelihood is to be evaluated shape : (size_minibatch, D)
    :param model: a VAE
    :param samples : number of importance samples per example
    :return: the ll
    """
    model.eval()
    with torch.no_grad():
        # Get the means and sigma
        out, mean, logv = model(batch)
        sigma = torch.exp(.5 * logv) + .1e-8
        m = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        samples_value_normal = torch.randn((samples, batch.size()[0], mean.size()[1]))

        # Get the importance weights
        proba_q = m.log_prob(samples_value_normal)
        aggregated_q = proba_q.sum(dim=2)

        # Get the hidden values and their prior proba
        value_z = mean + samples_value_normal * sigma
        proba_prior_z = m.log_prob(value_z)
        aggregated_z = proba_prior_z.sum(dim=2)

        # Forward pass these hidden values and get the probability of the observed values using the model
        cond_proba = list()
        for i in range(value_z.size()[1]):
            # Do this for each original points, for all samples
            estimated_conditional_point = model.decode(value_z[:, i, :]).view(-1, 784)

            # Compute the conditional likelihood of this image
            x = batch[i].view(-1, 784)
            proba_conditional = x * estimated_conditional_point + (1 - x) * (1 - estimated_conditional_point)
            aggregated_cond = proba_conditional.log().sum(dim=1)

            cond_proba.append(aggregated_cond)

        cond_proba_tensor = torch.stack(cond_proba).transpose(0, 1)
        proba_points = (aggregated_z + cond_proba_tensor - aggregated_q).exp()

        # Average over the samples
        proba_points.mean(dim=0)

        # Return result
    return proba_points


def main(model, test=False):
    valid_set = dataset.BinarizedMNIST(data_dir='../data/', split='valid', test=test)
    valid_loader = DataLoader(valid_set, batch_size=64, shuffle=True, num_workers=10)

    test_set = dataset.BinarizedMNIST(data_dir='../data/', split='test', test=test)
    test_loader = DataLoader(valid_set, batch_size=64, shuffle=True, num_workers=10)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    for i, batch in enumerate(valid_loader):
        batch.to(device)
        prob = ll_importance_sample_batch(batch, model, samples=200)
        print(prob.shape)


if __name__ == '__main__':
    model = models.vae.VariationalAutoEncoder()
    model.load_state_dict(torch.load('vae.pth', map_location='cpu'))
    main(model, test=True)
