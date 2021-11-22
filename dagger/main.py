from train import *
import argparse

def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default='64')
    parser.add_argument('--lr', type=float, default='0.001')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--map',type=int, default=32)
    parser.add_argument('--train_steps', type=int, default=20)
    parser.add_argument('--clip_val',type=float, default=1.0)
    parser.add_argument('--epoch',type=int, default=100)
    parser.add_argument('--test_eps',type=int, default=10)
    parser.add_argument('--debug',type=bool, default=False)
    parser.add_argument('--loglevel',type=int, default=0)
    parser.add_argument('--transfer',type=int, default=3)
    parser.add_argument('--pre_trained',type=bool, default=True)

    args = parser.parse_args()

    return args
    
def main(config):
    train(config)
    
if __name__ == '__main__':
    config = parse_arg()
    main(vars(config))