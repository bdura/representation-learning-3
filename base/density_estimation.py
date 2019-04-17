#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:20:15 2019

@author: chin-weihuang
"""

from __future__ import print_function
import numpy as np
import torch
import matplotlib.pyplot as plt

from base.samplers import distribution3, distribution4
import models.estimators as estimators

from torch.optim import Adam

# plot p0 and p1
plt.figure()

# empirical
xx = torch.randn(10000)
f = lambda x: torch.tanh(x * 2 + 1) + x * 0.75
d = lambda x: (1 - torch.tanh(x * 2 + 1) ** 2) * 2 + 0.75
plt.hist(f(xx), 100, alpha=0.5, density=1)
plt.hist(xx, 100, alpha=0.5, density=1)
plt.xlim(-5, 5)

# exact
xx = np.linspace(-5, 5, 1000)
N = lambda x: np.exp(-x ** 2 / 2.) / ((2 * np.pi) ** 0.5)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy() ** (-1) * N(xx))
plt.plot(xx, N(xx))
plt.show()


def estimate_jsd(steps):
    js = estimators.JensenShannon(input_dimension=1)
    optimiser = Adam(js.parameters(), lr=.001)

    for step in range(steps):
        if not step % 20:
            print(step + 1)
        f_0 = torch.Tensor(next(distribution3(512)))
        f_1 = torch.Tensor(next(distribution4(512)))

        js.zero_grad()
        loss = js.loss(f_1, f_0)
        loss.backward()
        optimiser.step()

    return js


critic = estimate_jsd(2000)

############### plotting things
############### (1) plot the output of your trained discriminator
############### (2) plot the estimated density contrasted with the true density

xx = np.linspace(-5, 5, 1000)
torch_xx = torch.from_numpy(xx).float().view(-1, 1)
D_x = critic(torch_xx).detach().numpy()
f_0_x = N(xx)[:, np.newaxis]

estimated = f_0_x * D_x / (1 - D_x)
#
# plt.plot(xx, estimated)
# plt.plot(xx, N(xx))
# plt.show()

r = D_x
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(xx, r)
plt.title(r'$D(x)$')

estimate = estimated
plt.subplot(1, 2, 2)
plt.plot(xx, estimate)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy() ** (-1) * N(xx))
plt.legend(['Estimated', 'True'])
plt.title('Estimated vs True')
plt.show()
