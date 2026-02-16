from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_centroid_loss(centroids, labels):
    """
    TODO: special case for centroid loss
    """
    q = centroids.shape[1]  # self.args.n_bits
    centroids = F.normalize(centroids)  # B x K
    cos_sim = centroids @ centroids.T  # B x B
    p = torch.softmax(q**0.5 * cos_sim, dim=1)
    y = (labels @ labels.T > 0).float()
    loss = y * torch.log(p) + (1 - y) * torch.log(1 - p)
    loss = -torch.mean(loss)
    return loss


def calc_centroid_loss_v2(centroids, labels):
    """
    simple version of calc_centroid_loss
    """
    q = centroids.shape[1]  # self.args.n_bits
    centroids = F.normalize(centroids)  # B x K
    cos_sim = centroids @ centroids.T  # B x B
    p = torch.softmax(q**0.5 * cos_sim, dim=1)
    y = (labels @ labels.T > 0).float()
    loss = torch.nn.BCELoss()(p, y)
    return loss


class CenterHashingLikeLoss(nn.Module):
    """
    like CenterHashingLoss using centroid as proxies
    """

    def __init__(self, args: Namespace):
        super().__init__()
        self.args = args

        # store all labels
        self.Y = nn.Parameter(torch.randn(args.n_samples, args.nclass), requires_grad=False)
        # store all logits
        self.U = nn.Parameter(torch.randn(args.n_samples, args.hash_bit), requires_grad=False)

    def forward(self, logits, labels, centroids, index=None):
        L_C = self.calc_center_loss(logits, labels, centroids)
        if index is None:
            # local
            L_P = self.calc_pair_loss(logits, labels)
        else:
            # global
            self.U[index, :] = logits.detach()
            self.Y[index, :] = labels
            L_P = self.calc_pair_loss(logits, labels, self.U, self.Y)
        L_Q = self.calc_q_loss(logits)
        # print("L_C", L_C.item())
        # print("L_P", L_P.item())
        # print("L_Q", L_Q.item())
        # Eq. (20)
        loss = L_C + self.args.lambda1 * L_P + self.args.lambda2 * L_Q
        return loss

    def calc_pair_loss(self, logits, labels, ref_logits=None, ref_labels=None):
        """
        Loss for Similar Pairs of the Same Hash Center
        L_P in Eq. (17)
        """
        logits = F.normalize(logits)
        if ref_logits is None:
            ref_logits = logits
        else:
            ref_logits = F.normalize(ref_logits)
        if ref_labels is None:
            ref_labels = labels
        cos_sim = logits @ ref_logits.T
        mask = (labels @ ref_labels.T > 0).float()

        # only the positive pair
        # NOTE: here q=1 is diff from paper
        loss = torch.sum(mask * torch.log(1 + torch.exp((1 - cos_sim) / 2))) / torch.sum(mask)
        return loss

    def calc_q_loss(self, logits):
        """
        Quantization loss
        L_Q in Eq. (19)
        """
        loss = (logits.abs() - 1).pow(2).mean()
        return loss

    def calc_center_loss(self, logits, labels, centroids):
        """
        Loss towards Hash Centroids
        L_C in Eq. (14)
        """
        q = self.args.n_bits
        logits = F.normalize(logits)  # B x K
        centroids = F.normalize(centroids)  # C x K / centroids case: B x K
        cos_sim = logits @ centroids.T  # B x C / centroids case: B x B
        p = torch.softmax(q**0.5 * cos_sim, dim=1)
        y = (labels @ labels.T > 0).float()
        loss = y * torch.log(p) + (1 - y) * torch.log(1 - p)
        # loss2 = calc_centroid_loss(centers, labels)
        # loss = loss1 + loss2
        loss = -torch.mean(loss)
        return loss


if __name__ == "__main__":
    N = 5

    _args = Namespace(n_bits=16, n_classes=10, lambda1=1.0, lambda2=1e-4, n_samples=10 * N)
    _criterion = CenterHashingLikeLoss(_args)
    _logits = torch.randn(N, _args.n_bits).cuda()
    _targets = torch.randint(0, _args.n_classes, (N,)).cuda()
    _labels = F.one_hot(_targets, _args.n_classes).float()
    _centers = torch.randn(_args.n_classes, _args.n_bits).cuda()
    _centroids = torch.randn(N, _args.n_bits).cuda()

    # import pickle
    #
    # with open("temp.pkl", "ab") as f:
    #     pickle.dump({"logits": _logits, "labels": _labels, "centers": _centers}, f)

    # with open("temp.pkl", "rb") as f:
    #     data = pickle.load(f)

    print(_criterion(_logits, _labels, _centroids).item())

    # print(calc_centroid_loss(_centroids, _labels))
    # print(calc_centroid_loss_v2(_centroids, _labels))
