import argparse
import itertools
import os
import warnings
from sched import scheduler

import torch
from torch.optim import lr_scheduler

from model import ComSRE
from data_utils import *
from configure import get_default_config

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cpu', help='gpu device ids')
parser.add_argument('--print_num', type=int, default='100', help='gap of print evaluations')
parser.add_argument('--city', type=str, default='xa')

args = parser.parse_args()


def main():
    config = get_default_config()
    config['city'] = args.city
    config['print_num'] = args.print_num

    # Environments
    warnings.filterwarnings("ignore")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    use_cuda = torch.cuda.is_available()
    device = torch.device(args.device)

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    comsre = ComSRE(config)
    optimizer = torch.optim.Adam(
        itertools.chain(comsre.autoencoder_a.parameters(), comsre.autoencoder_s.parameters(),
                        comsre.autoencoder_d.parameters(),
                        comsre.W_k.parameters(), comsre.W_v.parameters(),
                        comsre.proj.parameters(),
                        ), lr=config[args.city]['lr'], weight_decay=1e-3)

    comsre.to_device(device)

    # Load data
    redata = RegionData(config)
    xs_raw = [redata.a_m, redata.s_m, redata.d_m]
    xs = []
    for view in range(len(xs_raw)):
        xs.append(torch.from_numpy(xs_raw[view]).float().to(device))

    # Training
    comsre.train(config, redata, xs, optimizer, device)



main()
