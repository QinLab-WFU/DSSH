import torch
import torch.nn as nn
from torch.nn.functional import normalize
import torch.nn.functional as F

def smooth_CE(logits, labels, peak):
    # logits - [batch, num_cls]
    # label - [batch]
    A = ((labels == 0).sum(dim=1) == labels.shape[1])
    labels[A == True] = 1
    batch, num_cls = logits.size()
    label_logits = labels
    smooth_label = torch.ones(logits.size()) * (1 - peak) / (num_cls - 1)
    smooth_label[label_logits == 1] = peak

    logits = F.log_softmax(logits, -1)
    ce = torch.mul(logits, smooth_label.to(logits.device))
    loss = torch.mean(-torch.sum(ce, -1))  # batch average

    return loss

class MultiSimilarityLoss(nn.Module):
    def __init__(self):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.1
        self.margin = 0.1

        self.scale_pos = 2.0
        self.scale_neg = 40.0

    def forward(self, feats, labels, feat2=None,dataset="MSLOSS"):
        assert feats.size(0) == labels.size(0)
        batch_size = feats.size(0)
        if feat2 is None:
            sim_mat = normalize(torch.matmul(feats, torch.t(feats)))
        else:
            sim_mat = normalize(torch.matmul(feats,torch.t(feat2)))

        if dataset == "cifar10-1":
            labels = torch.argmax(labels , dim=1)
        else:
            labels = labels @ labels.t() > 0

        epsilon = 1e-5
        loss = list()

        for i in range(batch_size):
            pos_pair_ = sim_mat[i][labels == labels[i] if dataset =='cifar10-1' else labels[i]]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            
            neg_pair_ = sim_mat[i][labels != labels[i] if dataset =='cifar10-1' else (labels[i] == False)]

            if torch.numel(pos_pair_) == 0 or torch.numel(neg_pair_) == 0:
                continue

            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_) ]
            pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_) ]

            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue
                    
            # weighting step
            pos_loss = 1.0 / self.scale_pos * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
            neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)

        loss = sum(loss) / batch_size
        return loss