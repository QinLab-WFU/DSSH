import torch
from torchvision import models
import torch.nn as nn

from stochman import laplace
from stochman.utils import convert_to_stochman
from stochman.nnj import L2Norm

from torch.nn.utils import parameters_to_vector, vector_to_parameters

class MultiChannelNet(nn.Module):
    def __init__(self, nbit = 16):
        super(MultiChannelNet,self).__init__()

        self.alpha = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.zeros(1), requires_grad=True)

        latent_dim = nbit

        self.backbone = models.resnet50(pretrained=True)
        # self.backbone = self.backbone[:-1]
        # self.hash_fc = nn.Sequential(
        #     nn.Linear(1000,512),
        #     nn.BatchNorm1d(512),
        #     nn.Linear(512,nbit),
        #     nn.Tanh()
        # )
        self.linear = nn.Sequential(nn.Linear(1000, latent_dim), L2Norm())

    def forward(self, x):

        x = self.backbone(x)
        x = self.linear(x)

        return x


if __name__ == "__main__":

    net = MultiChannelNet(nbit=16)

    input = torch.randn(32,3,224,224)

    output = net(input,n_samples=3)
    print(output['z_samples'].shape)

    # indice_tuple = minner(labels)
    # criterion(output['z_samples'],net.alpha,net.beta,indice_tuple)
