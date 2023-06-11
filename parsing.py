import argparse

def make_parser():
    parser = argparse.ArgumentParser(description='KUKA Simulation')
    parser.add_argument('--demo', type=int, default = 0)
    parser.add_argument('--tasknum', type=int, default = 11)
    parser.add_argument('--output_dim', type=int, default = 6)
    parser.add_argument('--w_save', type=int, default = 1)
    parser.add_argument('--w_reuse', type=int, default = 0)
    parser.add_argument('--learn', type=int, default = 1)
    parser.add_argument('--gain', type=float, default = 0.35)
    parser.add_argument('--gui', type=int, default = 1)
    parser.add_argument('--control', type=str, default = 'force')
    parser.add_argument('--simcount', type=int, default = 10000)
    parser.add_argument('--sfinit', type=float, default = 2)
    parser.add_argument('--visual', type=int, default = 0)
    parser.add_argument('--seqlen', type=int, default = 16)
    parser.add_argument('--pretrain', type=int, default = 0)
    parser.add_argument('--device', type=str, default = 'cuda:0')
    parser.add_argument('--baseline',type=str,default = 'DDP')
    args = parser.parse_args()
    return args