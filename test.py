import torch
from torchvision import datasets, transforms
from modules.base.config import create_config
from modules.visualize import save_histogram, save_variation

DATA_DIR = 'data'
LOG_DIR = 'test'


def main():
    run = 'test-vae-mse'
    experiment_dir = f'{LOG_DIR}/{run}'
    cfg = create_config(f'{experiment_dir}/config.yaml')

    # Dataset

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(DATA_DIR, train=False, transform=transforms.ToTensor()),
        batch_size=cfg.experiment.batch_size, shuffle=False)

    device: str = "cuda" if torch.cuda.is_available() else 'cpu'
    save_histogram(
        config_path=f'{experiment_dir}/config.yaml',
        dataset=test_loader.dataset,
        device=device,
        histogram_dir='outputs',
    )
    save_variation(
        f'{experiment_dir}/config.yaml',
        dataset=test_loader.dataset,
        device=device,
        fname=f'{run}.png',
    )


if __name__ == '__main__':
    main()
