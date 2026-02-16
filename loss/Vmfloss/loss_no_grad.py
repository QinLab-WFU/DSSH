import torch
import torch.nn.functional as F
from torch import nn

import utils
from PDML.von_mises_fisher import VonMisesFisher


class VMFSoftmax(nn.Module):
    """von Mises-Fisher softmax cross-entropy."""

    def __init__(self, n_classes, n_bits, n_samples, init_temp):
        """Initializes a VMFSoftmax object."""
        super().__init__()
        self.n_samples = n_samples
        self.dist = VonMisesFisher
        self.vmf_projection = utils.get_norm_method_by_name("vmf")
        self.l2_norm_method = utils.get_norm_method_by_name("l2")
        self.temp = nn.Parameter(torch.zeros(1, 1, dtype=torch.float32, requires_grad=True) + init_temp)

        self.proxies = nn.Parameter(torch.Tensor(n_classes, n_bits), requires_grad=True)
        nn.init.kaiming_normal_(self.proxies, mode="fan_out")

    def forward(self, embeddings, labels):
        mu_z, kappa_z = self.l2_norm_method(embeddings, return_norms=True)
        z_dist = self.dist(mu_z, kappa_z)

        mu_p, kappa_p = self.l2_norm_method(self.proxies, return_norms=True)
        p_dist = self.dist(mu_p, kappa_p)

        z_samples = z_dist.sample(torch.Size([self.n_samples]))  # n_samples x B x K
        p_samples = p_dist.sample(torch.Size([self.n_samples]))  # n_samples x C x K

        mat = 2 - 2 * torch.einsum("nbk,nck->nbc", z_samples, p_samples)  # n_samples x B x C

        # (n_samples x B).mean()
        # return torch.sum(-labels * F.log_softmax(-mat / self.temp, -1), -1).mean()
        return torch.sum(-labels * F.log_softmax(-mat, -1), -1).mean()
