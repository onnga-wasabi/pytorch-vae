import json
from tqdm import tqdm
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from .config import Config


@dataclass
class State:
    epoch: int
    iteration: int
    epoch_pbar: tqdm
    iteration_pbar: tqdm


class BaseTrainer(object):

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            network: nn.Module,
            data_loader: torch.utils.data.DataLoader,
            device: str,
            writer: SummaryWriter,
            log_dir: str,
            config: Config,
            val_data_loader: torch.utils.data.DataLoader = None,
    ):
        self.optimizer = optimizer
        self.network = network.to(device)
        self.data_loader = data_loader
        self.data_iter = iter(self.data_loader)
        self.device = device
        self.writer = writer
        self.log_dir = log_dir
        self.config = config

        if val_data_loader:
            self.val_data_loader = val_data_loader
            self.val_data_iter = iter(self.val_data_loader)

        # self.result = {'name': self.log_dir, 'train_scores': [], 'val_scores': []}
        self.extra_setup()
        self.state = State(
            epoch=0,
            iteration=0,
            epoch_pbar=tqdm(total=self.config.experiment.epoch, leave=False, ncols=50),
            iteration_pbar=tqdm(total=len(self.data_iter), leave=False),
        )

    def extra_setup(self):
        pass

    def save(self):
        self.network = self.network.to('cpu')
        torch.save(self.network.state_dict(), f'{self.log_dir}/model.pt')

    def fit(self):
        while self.state.epoch < self.config.experiment.epoch:
            self.update()

            self.network.eval()
            with torch.no_grad():
                if self.val_data_loader:
                    self.evaluate()
                self.new_epoch()
            self.network.train()

        self.state.epoch_pbar.close()
        with open(f'{self.log_dir}/result.json', 'w') as wf:
            json.dump(self.result, wf, indent=2)

        self.save()

    def update(self):
        results = []
        while True:
            try:
                self.optimizer.zero_grad()
                batch = next(self.data_iter)
                computed = self.compute(batch)
                computed['loss'].backward()
                self.optimizer.step()

                self.state.iteration += 1
                self.state.iteration_pbar.update(1)
                self.state.iteration_pbar.set_postfix(computed['log'])
                self.iteration_end(computed)

            except StopIteration:
                break
        self.update_end(results)

    def iteration_end(self, computed):
        for k, v in computed['log'].items():
            self.writer.add_scalar(f'Train/{k}', v, self.state.iteration)

    def update_end(self, results):
        pass

    def evaluate(self):
        results = []
        while True:
            try:
                batch = next(self.val_data_iter)
                computed = self.evaluate_func(batch)
                results.append(computed)
            except StopIteration:
                break
        self.evaluate_end(results)

    def evaluate_func(self, batch):
        return self.compute(batch)

    def evaluate_end(self, results):
        for k in results[0]['log'].keys():
            metric = np.mean([r['log'][k] for r in results])
            self.writer.add_scalar(f'Validation/{k}', metric, self.state.epoch)

    def new_epoch(self):
        self.epoch_end()
        self.state.epoch += 1
        self.state.epoch_pbar.update(1)
        self.state.iteration_pbar.close()
        self.state.iteration_pbar = tqdm(total=len(self.data_iter), leave=False)

        del(self.data_iter)
        self.data_iter = iter(self.data_loader)

        if self.val_data_iter:
            del(self.val_data_iter)
            self.val_data_iter = iter(self.val_data_loader)

    def epoch_end(self):
        """
        画像とかをhogehogeするなかここかな
        """
        pass

    def compute(self, batch):
        raise NotImplementedError
