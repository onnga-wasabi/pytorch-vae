import torch
import torch.nn.functional as F
from .base.trainer import BaseTrainer


class VAETrainer(BaseTrainer):

    def compute(self, batch):
        x, _ = batch
        x = x.to(self.device)
        recon_x, mu, logvar = self.network.forward(x)
        MSE = torch.mean(torch.sum(F.mse_loss(recon_x, x.view(-1, 784), reduction='none'), dim=1), dim=0)
        # MSE = torch.mean(torch.sum(F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='none'), dim=1), dim=0)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        print(mu.shape)
        KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
        loss = MSE + KLD
        return {
            'loss': loss,
            'log': {
                'Loss': loss.item(),
                'MSE': MSE.item(),
                'KLD': KLD.item(),
            }
        }

    def optimizer_setup(self):
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.experiment.learing_rate)

    def extra_setup(self):
        for x, _ in self.val_data_loader:
            self.visualize_val_data = x[:16]

    def epoch_end(self):
        epoch = self.state.epoch
        x = self.visualize_val_data.to(self.device)
        reconstructed_images = self.network(x.view(-1, 784))[0].view(-1, 1, 28, 28).detach().cpu()
        comparison = torch.cat([self.visualize_val_data, reconstructed_images])
        self.writer.add_images("Reconstructed", comparison, global_step=epoch, dataformats="NCHW")


Trainers = {
    'vae': VAETrainer,
}
