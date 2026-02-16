import math
import torch
import torch.nn.functional as F
from torch import nn

from loss.Vmfloss import utils
from loss.Vmfloss.von_mises_fisher import VonMisesFisher


def calc_q_loss(embeddings):
    """
    Quantization loss
    L_Q in Eq. (19) of Center-Hashing
    """
    loss = (embeddings.abs() - 1).pow(2).mean()
    return loss


def calc_pair_loss(embeddings, labels):
    """
    Loss for Similar Pairs of the Same Hash Center
    L_P in Eq. (18) of Center-Hashing
    """
    mask = (labels @ labels.T > 0).float()

    embeddings = F.normalize(embeddings)
    cos_sim = embeddings @ embeddings.T

    # only the positive pair
    log_term = torch.log(1 + torch.exp(0.5 * (1 - cos_sim)))
    loss = torch.sum(mask * log_term) / torch.sum(mask)
    return loss


def vmf_class_weight_init(weights, kappa_confidence, embedding_dim):
    """Initializes class weight vectors as vMF random variables."""
    # This is the empirical approximation for initialization the vMF distributions
    # for each class in the final layer (Equation 19 in the paper).
    nn.init.normal_(
        weights,
        mean=0.0,
        std=(kappa_confidence / (1.0 - kappa_confidence**2)) * ((embedding_dim - 1.0) / math.sqrt(embedding_dim)),
    )


class VMFSoftmax(nn.Module):
    """von Mises-Fisher softmax cross-entropy."""

    def __init__(self, n_classes, n_bits, n_samples, init_temp, kappa_confidence):
        """Initializes a VMFSoftmax object."""
        super().__init__()
        self.n_bits = n_bits
        self.n_samples = n_samples
        self.dist = VonMisesFisher#特定概率分布
        self.vmf_projection = utils.get_norm_method_by_name("vmf")
        self.l2_norm_method = utils.get_norm_method_by_name("l2")

        self.temp = nn.Parameter(torch.zeros(1, 1, dtype=torch.float32, requires_grad=True) + init_temp)

        self.proxies = nn.Parameter(torch.Tensor(n_classes, n_bits), requires_grad=True)
        # nn.init.kaiming_normal_(self.proxies, mode="fan_out")
        vmf_class_weight_init(self.proxies, kappa_confidence, n_bits)

    def forward(self, embeddings, labels) -> torch.Tensor:
        """
        See Eq. 8
        L <= first_term(part1, part2) - second_term
        first_term = E_z{log[Σ_j(exp(part1-part2))]}
        z~ = embeddings
        z <- vMF.sample(μ=z~/||z~||, κ=||z~||)
        """
        beta = 1.0 / torch.exp(self.temp)

        # Little different from paper is that the input κ of log_C is squared
        log_C = utils.log_vmf_normalizer_approx

        exp_z, kappa_z = self.vmf_projection(embeddings, return_norms=True)
        exp_p_y = self.vmf_projection(labels @ self.proxies, return_norms=False)  # B x K
        second_term = beta * (exp_p_y * exp_z).sum(dim=1)
        # print(embeddings)
        mu_z = embeddings / kappa_z
        z_dist = self.dist(mu_z, kappa_z)
        # print(z_dist)

        # z.shape is n_samples x B x K -> B x n_samples x K
        z = z_dist.rsample(torch.Size([self.n_samples])).permute(1, 0, 2)

        p_tilde_norm2 = (self.proxies**2).sum(dim=1, keepdim=False)  # ||p~_j||_2^2, shape is C
        part1 = log_C(p_tilde_norm2, self.n_bits)

        # Since log_C operates on squared-kappa (κ^2),
        # we can simplify the argument in the denominator of the first term of the objective
        # ||p~j + βz||^2
        # = ||p~j|| + β^2 * ||z|| + 2 * β * z@p~j.T
        # = ||p~j|| + β^2 + 2 * β * z@p~j.T

        # B*n_samples x K, C x K -> B x n_samples x C
        batch_size = embeddings.size(0)
        z_dot_p_tilde = (z.reshape(-1, self.n_bits) @ self.proxies.T).view(batch_size, self.n_samples, -1)

        # shape in log_C: 1 x 1 x C + B x n_samples x C
        part2 = log_C(p_tilde_norm2.unsqueeze(0).unsqueeze(0) + beta**2 + 2.0 * beta * z_dot_p_tilde, self.n_bits)

        sub_term = part1.unsqueeze(0).unsqueeze(0) - part2
        first_term = torch.log(torch.exp(sub_term).sum(dim=2)).mean(dim=1)

        return (first_term - second_term).mean()
