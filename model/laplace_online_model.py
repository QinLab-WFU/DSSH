import torch
import torch.nn as nn
import os
import sys

sys.path.append("../stochman")
from stochman import nnj
from stochman import ContrastiveHessianCalculator, ArccosHessianCalculator
from stochman.laplace import DiagLaplace
from stochman.utils import convert_to_stochman
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm import tqdm
import wandb

from miners.custom_miners import TripletMarginMinerPR
from miners.triplet_miners import TripletMarginMiner

hessian_calculators = {
    "contrastive": ContrastiveHessianCalculator,
    "arccos": ArccosHessianCalculator,
}


class LaplaceOnlineModel(nn.Module):
    def __init__(self, args, savepath, seed):
        super().__init__(args, savepath, seed)

        self.max_pairs = args.max_pairs

        # if arccos, then remove normalization layer from model
        if self.args.loss == "arccos":
            self.model.linear = convert_to_stochman(self.model.linear[:-1])
        else:
            self.model.linear = convert_to_stochman(self.model.linear)

        self.hessian_calculator = hessian_calculators[args.loss](
            wrt="weight",
            shape="diagonal",
            speed="half",
            method=args.loss_approx,
        )

        self.laplace = DiagLaplace()

        self.dataset_size = args.dataset_size
        hessian = self.laplace.init_hessian(
            args.get("init_hessian", self.dataset_size), self.model.linear, "cuda:0"
        )
        self.scale_hs = args.get("scale_hessian", self.dataset_size ** 2)
        self.register_buffer("hessian", hessian)
        self.prior_prec = torch.tensor(1, device="cuda:0")

        # self.n_step_without_hessian_update = args.get("n_step_without_hessian_update", 0)
        # self.n_step_to_introduce_hessian = args.get("n_step_to_introduce_hessian", 0)
        # self.hessian_step_counter = 0
        self.hessian_memory_factor = args.hessian_memory_factor

        # hessian miners
        if self.place_rec:
            self.hessian_miner = TripletMarginMinerPR(
                margin=args.margin,
                collect_stats=True,
                type_of_triplets=args.get("type_of_triplets_hessian", "all"),
                posDistThr=args.get("posDistThr", 10),
                negDistThr=args.get("negDistThr", 25),
                distance=self.distance,
            )
        else:
            self.hessian_miner = TripletMarginMiner(
                margin=args.margin,
                collect_stats=True,
                distance=self.distance,
                type_of_triplets=args.get("type_of_triplets_hessian", "all"),  # [easy, hard, semihard, all]
            )

    def forward(self, x, n_samples=1):

        x = self.model.backbone(x)
        if hasattr(self.model, "pool"):
            x = self.model.pool(x)

        # get mean and std of posterior
        mu_q = parameters_to_vector(self.model.linear.parameters()).unsqueeze(1)

        # forward n times
        zs = []
        for i in range(n_samples):
            # use sample i that was generated in beginning of evaluation
            net_sample = self.nn_weight_samples[i]

            # replace the network parameters with the sampled parameters
            vector_to_parameters(net_sample, self.model.linear.parameters())

            z = self.model.linear(x)

            # ensure that we are on unit sphere
            z = z / z.norm(dim=-1, keepdim=True)

            zs.append(z)

        zs = torch.stack(zs)

        # compute statistics
        z_mu = zs.mean(dim=0)
        z_sigma = zs.std(dim=0)

        # put mean parameters back
        vector_to_parameters(mu_q, self.model.linear.parameters())

        return {"z_mu": z_mu, "z_sigma": z_sigma, "z_samples": zs.permute(1, 0, 2)}
