import os
from matplotlib.pyplot import hist
import numpy as np
from numpy.lib.histograms import histogram
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from .util import load_network


@torch.no_grad()
def create_histgram_values(network: nn.Module, dataset: np.array, device: str = 'cpu') -> np.array:
    network.to(device)
    latents = []

    for batch in dataset:
        x = batch
        x = np.array([x.numpy()])
        x = torch.from_numpy(x).flatten()
        x = x.to(device)
        latent = network.get_latent(x).to("cpu").numpy()
        latents.append(latent.flatten())

    del network

    latents = np.array(latents)
    return latents.flatten()


@torch.no_grad()
def create_variation(network, dataset, device) -> plt.figure:
    network.to(device)

    x = dataset[0]
    data = x.to(device)
    reconstructed = []

    noises = [-3, -2, -1, -.5, 0, .5, 1, 2, 3]

    for i in range(10):
        for noise in noises:
            z = network.get_latent(data.view(data.size(0), -1))
            z = torch.zeros_like(z)

            z[:, i] += noise

            x_hat = network.decode(z).to('cpu')
            reconstructed.append(x_hat)

    _, ax = plt.subplots(10, 9, figsize=(10, 10))
    for i, image in enumerate(reconstructed):
        j = i // 9
        k = i % 9
        image = image.numpy()
        img = image.reshape((28, 28))
        ax[j, k].imshow(img, cmap='gray')

        if j == 0:
            ax[j, k].set_title(f'{noises[k]}')

        ax[j, k].axis('off')

    plt.tight_layout()
    return plt


def save_histogram(config_path, dataset, device, histogram_dir):
    network = load_network(config_path)
    latent_values = create_histgram_values(network, dataset, device)

    os.makedirs(histogram_dir, exist_ok=True)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(latent_values, bins=1000)
    ax.set_title('Histogram')
    ax.set_xlabel('Real value of latent nodes')
    ax.set_ylabel('Freq of real value')

    fname = f"{histogram_dir}/hist-all.png"
    plt.savefig(fname)
    plt.close()


def save_variation(config_path, dataset, device, fname):
    network = load_network(config_path)
    variation_plt = create_variation(network, dataset, device)
    variation_plt.savefig(fname)
    variation_plt.close()
