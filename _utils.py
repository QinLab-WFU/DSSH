import os
import pickle

import math
import numpy as np
import torch
from loguru import logger
from openpyxl import Workbook
from openpyxl.reader.excel import load_workbook
from pytorch_metric_learning import losses, distances
from tqdm import tqdm
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import scipy.io as scio

def sampling(net, laplace, hessian_i, hessian_t, n_samples):

    # get mean and std of posterior
    mu_q_i = parameters_to_vector(net.image_linear.parameters()).unsqueeze(1)
    mu_q_t = parameters_to_vector(net.text_linear.parameters()).unsqueeze(1)
    sigma_q_i = laplace.posterior_scale(torch.relu(hessian_i))
    sigma_q_t = laplace.posterior_scale(torch.relu(hessian_t))

    # draw samples
    samples_i = laplace.sample(mu_q_i, sigma_q_i, n_samples)
    samples_t = laplace.sample(mu_q_t, sigma_q_t, n_samples)
    return samples_i, samples_t


def configure_metric_loss(loss, distance, margin):
    if distance == "cosine":
        dist = distances.CosineSimilarity()
    elif distance == "euclidean":
        dist = distances.LpDistance()

    if loss == "triplet":
        criterion = losses.TripletMarginLoss(margin=margin, distance=dist)
    elif loss in ("contrastive", "arccos"):
        pos_margin = margin if distance == "dot" else 0
        neg_margin = 0 if distance == "dot" else margin

        criterion = losses.ContrastiveLoss(
            pos_margin=pos_margin, neg_margin=neg_margin, distance=dist
        )
    else:
        raise NotImplementedError

    return criterion


def prediction_v2(net, dataloader, samples_i, samples_t, hash_bit, length):
    device = next(net.parameters()).device
    img_buffer = torch.empty(length, hash_bit, dtype=torch.float).to(device)
    text_buffer = torch.empty(length, hash_bit, dtype=torch.float).to(device)
    net.eval()
    logger.info(f'predicting({len(dataloader.dataset)})...')
    for image, text, label, idx in tqdm(dataloader):
        with torch.no_grad():
            x_i, x_t = net.forward(image.to(device), text.to(device), 1)
            mu_q_i = parameters_to_vector(net.image_linear.parameters()).unsqueeze(1)
            mu_q_t = parameters_to_vector(net.text_linear.parameters()).unsqueeze(1)

            zs_i = []
            zs_t = []
            for sample_i, sample_t in zip(samples_i, samples_t):

                vector_to_parameters(sample_i, net.image_linear.parameters())
                vector_to_parameters(sample_t, net.text_linear.parameters())
                z_i, z_t = net.linear(x_i, x_t)
                z_i = z_i / z_i.norm(dim=-1, keepdim=True)
                zs_i.append(z_i)
                zs_t.append(z_t)

            zs_i = torch.stack(zs_i)
            zs_t = torch.stack(zs_t)
            z_mu_i = torch.sign(zs_i.mean(dim=0))
            z_mu_t = torch.sign(zs_t.mean(dim=0))
            z_sigma_i = zs_i.std(dim=0)
            z_sigma_t = zs_t.std(dim=0)
            # print(z_mu.shape)

            vector_to_parameters(mu_q_i, net.image_linear.parameters())
            vector_to_parameters(mu_q_t, net.text_linear.parameters())

        img_buffer[idx, :] = z_mu_i.data
        text_buffer[idx, :] = z_mu_t.data
    return img_buffer, text_buffer


def prediction(net, dataloader, hash_bit, length):
    device = next(net.parameters()).device
    img_buffer = torch.empty(length, hash_bit, dtype=torch.float).to(device)
    text_buffer = torch.empty(length, hash_bit, dtype=torch.float).to(device)
    net.eval()
    logger.info(f'predicting({len(dataloader.dataset)})...')
    for image, text, label, idx in tqdm(dataloader):
        with torch.no_grad():
            image = image.to(device, non_blocking=True)
            text = text.to(device, non_blocking=True)
            x_i, x_t = net(image, text)
            x_i = torch.sign(x_i)
            x_t = torch.sign(x_t)
        img_buffer[idx, :] = x_i.data
        text_buffer[idx, :] = x_t.data
    return img_buffer, text_buffer


def calc_map_k(qB, rB, query_L, retrieval_L, k=None):
    num_query = len(query_L)
    if qB.is_cuda:
        qB = qB.cpu()
        rB = rB.cpu()
    map = 0
    if k is None:
        k = len(retrieval_L)
    for iter in range(num_query):
        gnd = (query_L[iter].unsqueeze(0).mm(retrieval_L.t()) > 0).type(torch.float).squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[iter, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float).to(gnd.device)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float) + 1.0
        map += torch.mean(count / tindex)
    map = map / num_query
    return map


def calc_hamming_dist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.t()))
    return distH


def save_mat(opt, query_img, query_txt, query_label, retrieval_img, retrieval_txt, retrieval_label, mAP, mode_name="i2t"):

    save_dir = os.path.join(opt.outf)
    os.makedirs(save_dir, exist_ok=True)

    query_img = query_img.cpu().detach().numpy()
    query_txt = query_txt.cpu().detach().numpy()
    retrieval_img = retrieval_img.cpu().detach().numpy()
    retrieval_txt = retrieval_txt.cpu().detach().numpy()
    query_labels = query_label.numpy()
    retrieval_labels = retrieval_label.numpy()

    result_dict = {
        'q_img': query_img,
        'q_txt': query_txt,
        'r_img': retrieval_img,
        'r_txt': retrieval_txt,
        'q_l': query_labels,
        'r_l': retrieval_labels
    }
    scio.savemat(os.path.join(save_dir, str(opt.hash_bit) + opt.dataset + "-" + mode_name+"-"+ str(mAP) + ".mat"), result_dict)