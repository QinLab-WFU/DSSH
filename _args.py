import argparse


def get_config():
    parser = argparse.ArgumentParser(description='train SCFR')
    parser.add_argument('--model', default='acmvh', help='model name')
    parser.add_argument('--loss', default='acmvh', help='loss')
    parser.add_argument('--scheduler', default=None, help='scheduler')
    parser.add_argument('--dataset', default='nuswide', choices=['flickr', 'nuswide', 'coco'])
    parser.add_argument('--hash_bit', type=int, default=16, help='length of hashing binary')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument("--clip-path", type=str, default="/home/yuebai/Data/Preload/ViT-B-32.pt", help="pretrained clip path.")

    parser.add_argument('--root', default='/home/yuebai/Data/Dataset/CrossModel/', help='path to dataset')
    parser.add_argument('--save', type=int, default=50, help='')

    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--test', type=int, default=1, help='period of test')
    parser.add_argument('--topk', default='-1', help='for calc map')

    parser.add_argument('--outf', default='checkpoints', help='folder to output model checkpoints')
    parser.add_argument('--checkpoint', type=int, default=50, help='checkpointing after batches')

    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument("--clip-lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.2)
    parser.add_argument("--warmup-proportion", type=float, default=0.1,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")

    # multi_model
    parser.add_argument("--max-words", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--query-num", type=int, default=5000)
    parser.add_argument("--train-num", type=int, default=10000)
    parser.add_argument("--lr-decay-freq", type=int, default=5)
    parser.add_argument("--display-step", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    # VmfLoss
    parser.add_argument("--lr_proxy", type=float, default=1e-2, help="learning rate of proxy")
    parser.add_argument("--lr_temp", type=float, default=1e-3, help="learning rate of temperature")
    parser.add_argument("--n_samples", type=int, default="10", help="number of samples")
    parser.add_argument("--init_temp", type=float, default=0.0, help="initial temperature")
    parser.add_argument("--kappa_confidence", type=float, default=0.7, help="kappa confidence")

    return parser.parse_args()
