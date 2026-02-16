import argparse
import math
import time
from copy import deepcopy

import os


import numpy as np
from loguru import logger
from torch import optim
# from pytorch_metric_learning.miners import TripletMarginMiner
from tqdm import tqdm

from _data import build_loader
from _init import _init_model, _init_dataset
from _utils import prediction, configure_metric_loss, sampling, calc_map_k, save_mat, prediction_v2
import json

import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from loss.MSLOSS import MultiSimilarityLoss, smooth_CE
from loss.Vmfloss.loss import calc_q_loss, calc_pair_loss
from loss.hib_loss import HibCriterion

from model.ResNet_LAM import MultiChannelNet

from _args import get_config


def sim_loss(image, text, label, alpha, threshold):
    index = label.sum(dim=1) > 1
    label_ = label[index].float()

    i_ = image[index]
    t_ = text[index]
    cos_sim = label_.mm(label_.T)

    if len((cos_sim == 0).nonzero()) == 0:
        reg_term_i = 0
        reg_term_t = 0
        reg_term_it = 0
    else:
        sim_i = F.normalize(i_, p=2, dim=1).mm(F.normalize(i_, p=2, dim=1).T)
        sim_t = F.normalize(t_, p=2, dim=1).mm(F.normalize(t_, p=2, dim=1).T)
        sim_it = F.normalize(i_, p=2, dim=1).mm(F.normalize(t_, p=2, dim=1).T)

        neg_i = alpha * F.relu(sim_i - threshold)
        neg_t = alpha * F.relu(sim_t - threshold)
        neg_it = alpha * F.relu(sim_it - threshold)

        reg_term_i = torch.where(cos_sim == 0, neg_i, torch.zeros_like(sim_i)).sum() / len((cos_sim == 0).nonzero())
        reg_term_t = torch.where(cos_sim == 0, neg_t, torch.zeros_like(sim_t)).sum() / len((cos_sim == 0).nonzero())
        reg_term_it = torch.where(cos_sim == 0, neg_it, torch.zeros_like(sim_it)).sum() / len((cos_sim == 0).nonzero())

    return reg_term_i + reg_term_t + reg_term_it

def run(opt, train_loader, test_loader, database_loader, qL, rL, logger):

    logger.info("init model.")
    net, criterion, optimizer, scheduler = _init_model(opt)

    count = 0
    total_time = 0
    best_i2t = 0.0
    best_t2i = 0.0
    best_epoch_i = 0.0
    best_epoch_t = 0.0

    for epoch in range(1, opt.epochs + 1):
        net.train()
        device = opt.device
        train_loss = 0
        # pbar = tqdm(train_loader, desc='Training', ascii=True, bar_format='{l_bar}{bar:10}{r_bar}')
        start_time = time.time()
        logger.info(f'training epoch:{epoch}/{opt.epochs}...')
        for image, text, label, idx in tqdm(train_loader):
            y = label.float().to(device)
            for i in range(y.shape[0]):
                if sum(y[i]) == 0:
                    rand = np.random.randint(0, 21)
                    y[i][rand] = 1
            if image.isnan().any() or y.isnan().any():
                print(image, y)
                print("-")
                exit()
            x_i, x_t = net(image.to(device), text.to(device))
            # MSLoss
            # i_loss = criterion(x_i, y)
            # t_loss = criterion(x_t, y)
            # it_loss = criterion(x_i, y, x_t)
            # ti_loss = criterion(x_t, y, x_i)
            # loss = i_loss + t_loss + it_loss + ti_loss

            # VMF
            loss_il = criterion(x_i, y)
            loss_tl = criterion(x_t, y)
            loss_it = criterion(torch.stack([x_i, x_t], dim=0).mean(dim=0), y)

            x_ii = x_i.mm(x_i.T)
            x_tt = x_t.mm(x_t.T)
            x_it = x_i.mm(x_t.T)
            yy = y.mm(y.T)
            loss_ce = torch.nn.CrossEntropyLoss()
            loss_sim = loss_ce(x_it, yy) + loss_ce(x_ii, yy) + loss_ce(x_tt, yy)

            loss = loss_il + loss_tl + loss_it
            loss += loss_sim
            # loss = loss_il + loss_tl + loss_sim

            # loss_sim = sim_loss(x_i, x_t, y, 0.8, 0.0)

            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        host_time = time.time() - start_time
        train_loss /= len(train_loader.dataset)

        total_time += host_time
        scheduler.step()

        logger.info(
            f"[Train][cost:{total_time:.4f}][dataset:{opt.dataset}][bits:{opt.hash_bit}][epoch:{epoch}/{opt.epochs}][train_loss:{train_loss}]")

        if epoch % opt.test == 0:
            encode_start_time = time.time()
            qB_i, qB_t = prediction(net, test_loader,  opt.hash_bit, opt.query_num)
            rB_i, rB_t = prediction(net, database_loader, opt.hash_bit, opt.num_database)
            encode_end_time = time.time() - encode_start_time

            mAPi2t = calc_map_k(qB_i, rB_t, qL, rL)
            mAPt2i = calc_map_k(qB_t, rB_i, qL, rL)
            # mAPi2i = calc_map_k(qB_i, rB_i, qL, rL)
            # mAPt2t = calc_map_k(qB_t, rB_t, qL, rL)

            logger.info(f"[Evaluation][cost:{encode_end_time:.4f}][dataset:{opt.dataset}][bits:{opt.hash_bit}][epoch:{epoch}/{opt.epochs}]")
            logger.info(f"[i->t:{mAPi2t:.5f}][t->i:{mAPt2i:.5f}]") #[i->i:{mAPi2i:.5f}][t->t:{mAPt2t:.5f}]")

            if mAPi2t <= best_i2t and mAPt2i <= best_t2i:
                count += 1
            else:
                count = 0

            if mAPi2t > best_i2t:
                best_i2t = mAPi2t
                best_epoch_i = epoch
                # torch.save(net.state_dict(), os.path.join(opt.outf, f"{epoch}-i2t:{mAPi2t}.pth"))
                # save_mat(opt, qB_i, qB_t, qL, rB_i, rB_t, rL, mAP=mAPi2t, mode_name="i2t")

            if mAPt2i > best_t2i:
                best_t2i = mAPt2i
                best_epoch_t = epoch
                # torch.save(net.state_dict(), os.path.join(opt.outf, f"{epoch}-t2i:{mAPt2i}.pth"))
                # save_mat(opt, qB_i, qB_t, qL, rB_i, rB_t, rL, mAP=mAPt2i, mode_name="i2t")

            logger.info(
                f"[best-i2t:{best_i2t:.5f}][epoch:{best_epoch_i}][best-t2i:{best_t2i:.5f}][epoch:{best_epoch_t}][count:{count}]")

            if count == 10 or epoch == opt.epochs:
                # torch.save(net.state_dict(), os.path.join(opt.outf, f"{epoch}-i2t:{mAPi2t}.pth"))
                # torch.save(net.state_dict(), os.path.join(opt.outf, f"{epoch}-t2i:{mAPt2i}.pth"))
                # save_mat(opt, qB_i, qB_t, qL, rB_i, rB_t, rL, mAP=mAPi2t, mode_name="i2t")
                # save_mat(opt, qB_i, qB_t, qL, rB_i, rB_t, rL, mAP=mAPt2i, mode_name="i2t")
                break

            # if epoch % opt.save == 0:
                # torch.save(net.state_dict(), os.path.join(opt.outf, f"{epoch}.pth"))
                # save_mat(opt, qB_i, qB_t, qL, rB_i, rB_t, rL, mode_name="i2t")
                # save_mat(opt, qB_i, qB_t, qL, rB_i, rB_t, rL, mode_name="t2i")

    return best_i2t, best_t2i, best_epoch_i, best_epoch_t


def feed_random_seed(seed=np.random.randint(1, 10000)):
    """feed random seed"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == '__main__':

    torch.cuda.empty_cache()#释放缓存分配器当前持有的所有未被占用的缓存内存
    opt = get_config()
    feed_random_seed()

    # batch_size is too small will cause "Too many open files..."
    torch.multiprocessing.set_sharing_strategy('file_system')

    dummy_logger_id = None
    rst = []
    for dataset in ['nuswide']:
        logger.info(f'processing dataset: {dataset}')
        opt.dataset = dataset

        logger.info("init dataset.")
        train_loader, test_loader, database_loader, train_label, query_label, retrieval_label = _init_dataset(opt)

        opt.num_train = len(train_label)
        # opt.query_num = len(query_labels)
        opt.num_database = len(retrieval_label)

        for hash_bit in [16, 32, 64, 128]:
            logger.info(f'processing hash-bit: {hash_bit}')
            opt.hash_bit = hash_bit

            opt.outf = f"./output/{dataset}/{hash_bit}"
            os.makedirs(opt.outf, exist_ok=True)

            if dummy_logger_id is not None:
                logger.remove(dummy_logger_id)
            dummy_logger_id = logger.add(f'{opt.outf}/train.log', rotation="500 MB", level="INFO")

            with open(f'{opt.outf}/config.json', 'w+') as f:
                json.dump(vars(opt), f, indent=4, sort_keys=True)

            best_i2t, best_t2i, best_epoch_i, best_epoch_t =\
            run(opt, train_loader, test_loader, database_loader, query_label, retrieval_label, logger)

            x = {
                "dataset": dataset,
                "hash_bit": hash_bit,
                "best_epoch_i": best_epoch_i,
                "best_epoch_t": best_epoch_t,
                "best_i2t": best_i2t,
                "best_t2i": best_t2i,
            }
            rst.append(x)

    for x in rst:
        print(
            f"[dataset:{x['dataset']}][hash-bit:{x['hash_bit']}][best-epoch-i:{x['best_epoch_i']}]"
            f"[best-epoch-t:{x['best_epoch_t']}][best_i2t:{x['best_i2t']:.4f}][best_t2i:{x['best_t2i']:.4f}]"
        )
