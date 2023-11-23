import os
import utils
import torch as t, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision as tv, torchvision.transforms as tr
import sys
import argparse
import numpy as np
from ExpUtils import *
from models.jem_models import F, CCF
from utils import plot, Hamiltonian, get_data
from models.jem_models import get_model_and_buffer



parser = argparse.ArgumentParser("LDA Energy Based Models")
parser.add_argument("--dataset", type=str, default="MVTec", choices=["MVTec", "cifar10", "svhn", "cifar100", 'tinyimagenet'])
parser.add_argument("--category", type=str, default="hazelnut")

# Patch slice size
parser.add_argument("--num_slices", type=int, default=8) # 이미지 슬라이스 크기(ex:8번 슬라이스하면 8x8 생성)
parser.add_argument("--img_size", type=int, default=256) # 불러올 이미지 사이즈

# optimization
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--decay_epochs", nargs="+", type=int, default=[60, 90, 120, 135], help="decay learning rate by decay_rate at these epochs")
parser.add_argument("--decay_rate", type=float, default=.2, help="learning rate decay multiplier")
parser.add_argument("--clf_only", action="store_true", default=False, help="If set, then only train the classifier")
# parser.add_argument("--labels_per_class", type=int, default=-1,
#                     help="number of labeled examples per class, if zero then use all labels")
parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--n_epochs", type=int, default=150)
parser.add_argument("--warmup_iters", type=int, default=-1,
                    help="number of iters to linearly increase learning rate, if -1 then no warmmup")
# loss weighting
parser.add_argument("--p_x_weight", type=float, default=1.)
parser.add_argument("--alpha", type=float, default=.1, help='reg_L weight')

# regularization
parser.add_argument("--dropout_rate", type=float, default=0.0)
parser.add_argument("--sigma", type=float, default=0.1, #3e-2,
                    help="stddev of gaussian noise to add to input, .03 works but .1 is more stable")
parser.add_argument("--weight_decay", type=float, default=4e-4)
# network
parser.add_argument("--norm", type=str, default='batch', choices=[None, "none", "batch", "instance", "layer", "act"], help="norm to add to weights, none works fine")
# EBM specific
parser.add_argument("--n_steps", type=int, default=10, help="number of steps of SGLD per iteration, 20 works for PCD")
parser.add_argument("--in_steps", type=int, default=5, help="number of steps of SGLD per iteration, 100 works for short-run, 20 works for PCD")
parser.add_argument("--width", type=int, default=10, help="WRN width parameter")
parser.add_argument("--depth", type=int, default=28, help="WRN depth parameter")
parser.add_argument("--uncond", action="store_true", default = False, help="If set, then the EBM is unconditional")
parser.add_argument("--class_cond_p_x_sample", action="store_true", default = False,
                    help="If set we sample from p(y)p(x|y), othewise sample from p(x),"
                            "Sample quality higher if set, but classification accuracy better if not.")
parser.add_argument("--buffer_size", type=int, default=100)
parser.add_argument("--reinit_freq", type=float, default=0.05)
parser.add_argument("--n_classes", type=int, default=1)

# SGLD or PYLD
parser.add_argument("--sgld_lr", type=float, default=0.0)
parser.add_argument("--sgld_std", type=float, default=0)
parser.add_argument("--pyld_lr", type=float, default=0.2)
# logging + evaluation
parser.add_argument("--save_dir", type=str, default='/app/outputs/experiment')
parser.add_argument("--dir_path", type=str, default='/app/outputs/experiment')
parser.add_argument("--log_dir", type=str, default='/app/outputs/runs')
parser.add_argument("--log_arg", type=str, default='JEMPP-n_steps-in_steps-pyld_lr-norm-plc')
parser.add_argument("--ckpt_every", type=int, default=15, help="Epochs between checkpoint save")
parser.add_argument("--eval_every", type=int, default=1, help="Epochs between evaluation")
parser.add_argument("--print_every", type=int, default=15, help="Iterations between print")
parser.add_argument("--load_path", type=str, default=None)
parser.add_argument("--print_to_log", action="store_true", help="If true, directs std-out to log file")
parser.add_argument("--plot_cond", action="store_true", default=False, help="If set, save class-conditional samples")
parser.add_argument("--plot_uncond", action="store_true", help="If set, save unconditional samples")
parser.add_argument("--n_valid", type=int, default=5000)

parser.add_argument("--plc", type=str, default="alltrain1", help="alltrain1, alltrain2, eval")

parser.add_argument("--eps", type=float, default=1, help="eps bound")
parser.add_argument("--model", type=str, default='yopo')
parser.add_argument("--novis", action="store_true", help="")
parser.add_argument("--debug", action="store_true", help="")
parser.add_argument("--exp_name", type=str, default="JEMPP", help="exp name, for description")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--gpu-id", type=str, default="0")
args = parser.parse_args()

args = parser.parse_args()

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

f, _ = get_model_and_buffer(args, device)

# print(f)

from torchsummary import summary

summary (f, (3, 32, 32))