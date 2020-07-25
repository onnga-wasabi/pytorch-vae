import torch
import torch.nn.functional as F

from .base.trainer import BaseTrainer
from .visualize import create_histgram_values, create_variation


class VAETrainer(BaseTrainer):

    def compute(self, batch):
        '''
        MSE = torch.mean(torch.sum(F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='none'), dim=1), dim=0)
        see Appendix B from VAE paper:
        Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        https://arxiv.org/abs/1312.6114
        0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        '''

        x, _ = batch
        x = x.to(self.device)
        recon_x, mu, logvar = self.network.forward(x)
        MSE = torch.mean(torch.sum(F.mse_loss(recon_x, x.view(-1, 784), reduction='none'), dim=1), dim=0)
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

    @torch.no_grad()
    def epoch_end(self):
        self.network.eval()

        latent_values = create_histgram_values(self.network, self.visualize_val_data, device='cuda:0')
        self.wandb.log({"Latent Values": self.wandb.Histogram(latent_values, num_bins=512)})

        variation_plt = create_variation(self.network, self.visualize_val_data, device='cuda:0')
        self.wandb.log({"Variation": variation_plt})
        variation_plt.close()

        x = self.visualize_val_data[:8]
        x = x.to(self.device)
        self.network.train()


Trainers = {
    'vae': VAETrainer,
}
