import torch
import torch.nn as nn


class HibCriterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z_samples, alpha, beta, indices_tuple):

        n_samples = z_samples.shape[1]

        if len(indices_tuple) == 3:
            a, p, n = indices_tuple
            ap = an = a
        elif len(indices_tuple) == 4:
            ap, p, an, n = indices_tuple

        alpha = torch.nn.functional.softplus(alpha)

        loss = 0
        for i in range(n_samples):
            z_i = z_samples[:, i, :]  ## [Batch , num_sample , bit] -> [Batch , bit]
            for j in range(n_samples):
                z_j = z_samples[:, j, :]  #

                prob_pos = torch.sigmoid(- alpha * torch.sum((z_i[ap] - z_j[p]) ** 2, dim=1) + beta) + 1e-6
                prob_neg = torch.sigmoid(- alpha * torch.sum((z_i[an] - z_j[n]) ** 2, dim=1) + beta) + 1e-6

                # maximize the probability of positive pairs and minimize the probability of negative pairs
                loss += -torch.log(prob_pos) - torch.log(1 - prob_neg)
        loss = loss / (n_samples ** 2)

        return loss.mean()


def get_matches_and_diffs(labels):
    matches = (labels.float() @ labels.float().T).byte()
    diffs = matches ^ 1  # 异或运算得到负标签的矩阵
    return matches, diffs


def get_all_triplets_indices_vectorized_method(all_matches, all_diffs):
    """
    Args:
        all_matches (torch.Tensor): 相同标签
        all_diffs (torch.Tensor): 不相同标签

    Processing : all_matches.unsqueeze(2) -> [Batch,Batch,1]
                 all_diffs.unsqeeeze(1) -> [Batch,1,Batch]

    Returns:
        torch.Tensor: _description_
    """

    triplets = all_matches.unsqueeze(2) * all_diffs.unsqueeze(1)
    return torch.where(triplets)


class TripletMinner(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sim_mat = get_matches_and_diffs
        self.selctor = get_all_triplets_indices_vectorized_method

    def forward(self, labels):
        a, b = self.sim_mat(labels)
        c = self.selctor(a, b)

        return c


# minner = TripletMinner()

# indice_tuple = minner(label)