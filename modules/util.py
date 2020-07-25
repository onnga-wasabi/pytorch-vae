import torch
import torch.nn as nn
from .base.config import create_config
from .networks import Networks


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
