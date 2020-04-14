import os
import shutil

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from networks import Networks
from trainers import Trainers
from utils.config import Config, create_config


def check_assertions(cfg: Config) -> None:
    assert cfg.model.network in Networks.keys()
    assert cfg.experiment.updater in Trainers.keys()
    # assert cfg.name not in os.listdir('./runs')


def main():
    cfg = create_config('./config.yaml')
    check_assertions(cfg)

    log_dir = './runs'
    experiment_dir = f'{log_dir}/{cfg.name}'
    writer = SummaryWriter(log_dir=f'{experiment_dir}')
    shutil.copy('./config.yaml', f'{experiment_dir}/config.yaml')

    # Dataset
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=cfg.experiment.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
        batch_size=cfg.experiment.batch_size, shuffle=False)
    network = Networks[cfg.model.network]()
    trainer = Trainers[cfg.experiment.updater](
        network,
        train_loader,
        writer,
        experiment_dir,
        cfg,
        test_loader,
    )
    # Learning
    trainer.fit()
    writer.close()


if __name__ == '__main__':
    main()
