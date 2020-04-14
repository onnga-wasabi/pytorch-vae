import os
import shutil

import torch
from torchvision import datasets, transforms

from networks import Networks
from trainers import Trainers
from utils.config import Config, create_config

LOG_DIR = '.runs'
DATA_DIR = 'data'
CONFIG_NAME = 'config.yaml'


def check_assertions(cfg: Config) -> None:
    assert cfg.model.network in Networks.keys()
    assert cfg.experiment.updater in Trainers.keys()
    assert cfg.name not in os.listdir('./runs')


def main():
    cfg = create_config(CONFIG_NAME)
    check_assertions(cfg)

    experiment_dir = f'{LOG_DIR}/{cfg.name}'
    shutil.copy(CONFIG_NAME, f'{experiment_dir}/{CONFIG_NAME}')

    # Model
    network = Networks[cfg.model.network]()

    # Dataset
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(DATA_DIR, train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=cfg.experiment.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(DATA_DIR, train=False, transform=transforms.ToTensor()),
        batch_size=cfg.experiment.batch_size, shuffle=False)

    trainer = Trainers[cfg.experiment.updater](
        cfg,
        network,
        train_loader,
        experiment_dir,
        test_loader,
    )

    # Learning
    trainer.fit()


if __name__ == '__main__':
    main()
