from train import *
import argparse

def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default='64')
    parser.add_argument('--lr', type=float, default='0.001')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--load_prop', type=float, default=0.1)
    parser.add_argument('--load_last_ckpt', type=bool, default=False)
    parser.add_argument('--last_ckpt_dir', type=str, default=None)
    args = parser.parse_args()

    return args


def main(config):
    train(config)
    
if __name__ == '__main__':
    config = parse_arg()
    main(vars(config))