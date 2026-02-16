import os

import math
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import loss.losses
from _args import get_config
from dataset.dataloader import dataloader
from loss.MSLOSS import MultiSimilarityLoss
from loss.VPTSPLoss import CenterHashingLikeLoss
from loss.Vmfloss.loss import VMFSoftmax
from model.hash_model import LAM
from model.optimization import BertAdam


def _init_dataset(opt):
    if opt.dataset == 'flickr':
        data_dir = os.path.join(opt.root, 'MIRFLICKR-25K')
        opt.nclass = 24
    elif opt.dataset == 'coco':
        data_dir = os.path.join(opt.root, 'MS-COCO')
        opt.nclass = 80
    elif opt.dataset == 'nuswide':
        data_dir = os.path.join(opt.root, 'NUS-WIDE')
        opt.nclass = 21
    elif opt.dataset == 'iapr':
        data_dir = os.path.join(opt.root, 'IAPR')
        opt.nclass = 291
    else:
        raise ValueError("Unknown dataset")
    index_file = os.path.join(data_dir, "index.mat")
    if opt.dataset == 'nuswide':
        caption_file = os.path.join(data_dir, "caption.txt")
    else:
        caption_file = os.path.join(data_dir, "caption.mat")
    label_file = os.path.join(data_dir, "label.mat")
    train_data, query_data, retrieval_data = dataloader(captionFile=caption_file,
                                                        indexFile=index_file,
                                                        labelFile=label_file,
                                                        maxWords=opt.max_words,
                                                        imageResolution=opt.resolution,
                                                        query_num=opt.query_num,
                                                        train_num=opt.train_num,
                                                        seed=opt.seed)
    train_labels = train_data.get_all_label()
    query_labels = query_data.get_all_label()
    retrieval_labels = retrieval_data.get_all_label()

    # logger.info(f"query shape: {query_labels.shape}")
    # logger.info(f"retrieval shape: {retrieval_labels.shape}")
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=True,
        shuffle=True
    )
    query_loader = DataLoader(
        dataset=query_data,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=True,
        shuffle=True
    )
    retrieval_loader = DataLoader(
        dataset=retrieval_data,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=True,
        shuffle=True
    )

    return train_loader, query_loader, retrieval_loader, train_labels, query_labels, retrieval_labels


def _init_model(opt):
    device = opt.device
    HashModel = LAM
    # opt.n_samples = len(train_loader.dataset)
    opt.n_samples = opt.num_train
    net = HashModel(num_class=opt.nclass, outputDim=opt.hash_bit, clipPath=opt.clip_path).to(device)

    net.float()

    # criterion = MultiSimilarityLoss().to(device)
    # criterion = loss.losses.configure_metric_loss("contrastive", "euclidean", 0.7)
    # criterion = nn.CrossEntropyLoss()
    criterion = VMFSoftmax(opt.nclass, opt.hash_bit, opt.n_samples, opt.init_temp, opt.kappa_confidence).to(device)
    # criterion = CenterHashingLikeLoss(opt).to(device)

    to_optim = [
        {'params': net.clip.parameters(), 'lr': opt.clip_lr},
        {'params': net.image_hash.parameters(), 'lr': opt.lr},
        {'params': net.text_hash.parameters(), 'lr': opt.lr},
        {"params": criterion.proxies, "lr": opt.lr_proxy},
        {"params": criterion.temp, "lr": opt.lr_temp},
    ]

    optimizer = BertAdam(params=to_optim, lr=opt.lr, warmup=opt.warmup_proportion, schedule='warmup_cosine',
                         b1=0.9, b2=0.98, e=1e-6, t_total=opt.num_train * opt.epochs,
                         weight_decay=opt.weight_decay, max_grad_norm=1.0)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=math.exp(-0.01))

    print(net)

    return net, criterion, optimizer, scheduler


def init_optimizer(optim_type, parameters, **kwargs):
    # optimizer_names = ["Adam", "RMSprop", "SGD"]
    # optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    if optim_type == "adam":
        optimizer = optim.Adam(parameters, **kwargs)
    elif optim_type == "rmsprop":
        optimizer = optim.RMSprop(parameters, **kwargs)
    elif optim_type == "adamw":
        optimizer = optim.AdamW(parameters, **kwargs)
    elif optim_type == "sgd":
        optimizer = optim.SGD(parameters, **kwargs)
    else:
        raise NotImplementedError(f"not support: {optim_type}")
    return optimizer


if __name__ == '__main__':
    # net = LAM(outputDim=16, clipPath="/home/yuebai/Data/Preload/ViT-B-32.pt").cuda()
    # state = torch.load("./output/coco/16/200.pth")
    # hessian_i = state.pop('hessian_i')
    # hessian_t = state.pop('hessian_t')
    # for key in state.keys():
    #     print(key)

    # criterion = MultiSimilarityLoss()
    # print(criterion.parameters)

    # optimizer = BertAdam([
    #     {'params': net.clip.parameters(), 'lr': 0.001},
    #     {'params': net.image_linear.parameters(), 'lr': 0.001},
    #     {'params': net.text_linear.parameters(), 'lr': 0.001},
    # ], lr=0.01, warmup=0.1, schedule='warmup_cosine',
    #     b1=0.9, b2=0.98, e=1e-6, t_total=10000 * 200,
    #     weight_decay=0.2, max_grad_norm=1.0)

    opt = get_config()
    opt.dataset = 'nuswide'
    train_loader, test_loader, database_loader, train_label, query_label, retrieval_label = _init_dataset(opt)
    print(query_label.shape)
    # root = f'/home/yuebai/Data/Dataset/CrossModal/NUS-WIDE-TC21'
    # import scipy.io as scio
    # indexs = scio.loadmat(f'/home/yuebai/Data/Dataset/CrossModal/NUS-WIDE/index.mat')["index"]
    #
    # indexs_ = []
    # for ind in indexs:
    #     idx = ind.strip().split('/')
    #     idx_ = idx[:8]
    #     idx_.append(idx[9])
    #     idx = '/'.join(idx_)
    #     indexs_.append(idx)
    #
    #
    # idx = {'index':indexs_}
    # scio.savemat(f'/home/yuebai/Data/Dataset/CrossModal/NUS-WIDE/index.mat', idx)
    print('ok')
