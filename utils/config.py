import yaml
from dataclasses import dataclass, fields


@dataclass
class ExperimentConfig:
    epoch: int
    batch_size: int
    learing_rate: float
    updater: str


@dataclass
class ModelConfig:
    network: str


@dataclass
class SparseConifg:
    ro: float
    beta: float
    fcwd: float


@dataclass
class ExternalConfig:
    sparse: SparseConifg


@dataclass
class Config:
    name: str
    experiment: ExperimentConfig
    model: ModelConfig
    external_config: ExternalConfig = None

    def __name__(self):
        return self.name


def load_from_dict(class_, dict_):
    try:
        fieldtypes = {f.name: f.type for f in fields(class_)}
        return class_(**{f: load_from_dict(fieldtypes[f], dict_[f]) for f in dict_})
    except TypeError:
        return dict_


def create_config(config_path: str) -> Config:
    with open(config_path, "r") as rf:
        config_data = yaml.safe_load(rf)
    config = load_from_dict(Config, config_data)
    return config
