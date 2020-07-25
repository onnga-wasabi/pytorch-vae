import os
import shutil

import torch
from torchvision import datasets, transforms

from modules.networks import Networks
from modules.trainers import Trainers
from modules.base.config import Config, create_config

LOG_DIR = 'test'
DATA_DIR = 'data'
CONFIG_NAME = 'config.yaml'
WANDB_PROJECT = 'pytroch-vae'


def check_assertions(cfg: Config) -> None:
    assert cfg.model.network in Networks.keys()
    assert cfg.experiment.updater in Trainers.keys()
    assert cfg.name not in os.listdir('./runs')


def main():
    cfg = create_config(CONFIG_NAME)
    check_assertions(cfg)

    experiment_dir = f'{LOG_DIR}/{cfg.name}'
    os.makedirs(experiment_dir, exist_ok=True)
    shutil.copy(CONFIG_NAME, f'{experiment_dir}/{CONFIG_NAME}')

    # Model
    network = Networks[cfg.model.network](cfg)

    # Dataset
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(DATA_DIR, train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=cfg.experiment.batch_size,
        shuffle=True,
        drop_last=True,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(DATA_DIR, train=False, transform=transforms.ToTensor()),
        batch_size=cfg.experiment.batch_size, shuffle=False)

    trainer = Trainers[cfg.experiment.updater](
        cfg,
        network,
        train_loader,
        experiment_dir,
        test_loader,
        WANDB_PROJECT,
    )

    # Learning
    trainer.fit()


if __name__ == '__main__':
    main()
