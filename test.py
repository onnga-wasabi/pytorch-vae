import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from modules.base.config import create_config
from modules.networks import Networks

import matplotlib.pyplot as plt

DATA_DIR = 'data'
LOG_DIR = 'test'


def get_logdir(config_path: str):
    logdir = '/'.join(config_path.split('/')[:-1])
    return logdir


def load_network(config_path: str) -> nn.Module:
    cfg = create_config(config_path)
    weights_path = get_logdir(config_path) + "/model_best.pt"
    network: nn.Module = Networks[cfg.model.network](cfg)
    network.load_state_dict(torch.load(weights_path), strict=False)
    network.eval()
    return network


@torch.no_grad()
def create_histgram(config_path: str, dataset: np.array, device: str = 'cpu') -> None:
    print('creating histogram...')
    network = load_network(config_path)
    network.to(device)
    latents = []
    histogram_dir = "outputs"
    os.makedirs(histogram_dir, exist_ok=True)

    for i, batch in enumerate(dataset):
        x, _ = batch
        x = np.array([x.numpy()])
        x = torch.from_numpy(x).flatten()
        x = x.to(device)
        latent = network.get_latent(x).to("cpu").numpy()
        # mu, logvar = network.get_mu_var(x)
        # mu = mu.to("cpu").numpy().flatten()
        # logvar = logvar.to("cpu").numpy().flatten()
        # mus.append(mu)
        # vars.append(np.exp(0.5 * logvar))
        latents.append(latent.flatten())

        # if i < 8:
        #     fname = f"{histogram_dir}/hist-{i}.png"
        #     fig = plt.figure(figsize=(10, 8))
        #     ax = fig.add_subplot(1, 1, 1)
        #     ax.hist(latent.flatten(), bins=1000)
        #     ax.set_title('Histogram')
        #     ax.set_xlabel('Real value of latent nodes')
        #     ax.set_ylabel('Freq of real value')
        #     plt.savefig(fname)
        #     plt.close()

    del network

    for i in range(20):
        if 1:
            fname = f"{histogram_dir}/hist-{i}_z.png"
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(1, 1, 1)
            mu = [m[i] for m in latents]
            ax.hist(mu, bins=1000)
            ax.set_title('Histogram mu')
            ax.set_xlabel('Real value of latent nodes')
            ax.set_ylabel('Freq of real value')
            plt.savefig(fname)
            plt.close()

            # fname = f"{histogram_dir}/hist-{i}_var.png"
            # fig = plt.figure(figsize=(10, 8))
            # ax = fig.add_subplot(1, 1, 1)
            # var = [v[i] for v in vars]
            # ax.hist(var, bins=1000)
            # ax.set_title('Histogram var')
            # ax.set_xlabel('Real value of latent nodes')
            # ax.set_ylabel('Freq of real value')
            # plt.savefig(fname)
            # plt.close()

    latents = np.array(latents)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(latents.flatten(), bins=1000)
    ax.set_title('Histogram')
    ax.set_xlabel('Real value of latent nodes')
    ax.set_ylabel('Freq of real value')
    fname = f"{histogram_dir}/hist-all.png"
    plt.savefig(fname)
    plt.close()


@torch.no_grad()
def validate_vae(config_path, dataset, device, fname):
    network = load_network(config_path)
    network.to(device)

    # batch = dataset[8]
    x, _ = dataset[8]
    data = x.to(device)
    reconstructed = []

    noises = [-2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2]

    for i in range(10):
        j = i
        # i = i + 50
        for noise in noises:
            z = network.get_latent(data.view(data.size(0), -1))
            z = torch.zeros_like(z)

            z[:, i] += noise

            x_hat = network.decode(z).to('cpu')
            print(x_hat.shape)
            reconstructed.append(x_hat)

    _, ax = plt.subplots(10, 9, figsize=(10, 10))
    for i, voxel in enumerate(reconstructed):
        j = i // 9
        k = i % 9
        voxel = voxel.numpy()
        # img = np.flip(voxel.transpose((1, 2, 0))[slice_idx], 0)
        img = voxel.reshape((28, 28))
        ax[j, k].imshow(img, cmap='gray')

        if j == 0:
            ax[j, k].set_title(f'{noises[k]}')

        ax[j, k].axis('off')

    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def main():
    run = 'test-vae-mse'
    experiment_dir = f'{LOG_DIR}/{run}'
    cfg = create_config(f'{experiment_dir}/config.yaml')

    # Dataset

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(DATA_DIR, train=False, transform=transforms.ToTensor()),
        batch_size=cfg.experiment.batch_size, shuffle=False)

    device: str = f"cuda" if torch.cuda.is_available() else 'cpu'
    # create_histgram(config_path=f'{experiment_dir}/config.yaml', dataset=test_loader.dataset, device=device)

    validate_vae(f'{experiment_dir}/config.yaml', dataset=test_loader.dataset, device=device, fname=f'{run}.png')


if __name__ == '__main__':
    main()
