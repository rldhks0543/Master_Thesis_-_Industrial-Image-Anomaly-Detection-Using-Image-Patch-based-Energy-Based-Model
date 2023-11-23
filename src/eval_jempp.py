# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
import matplotlib.pyplot as plt
import numpy as np
from MVTec import MVTEC

# Sampling
from tqdm import tqdm
t.backends.cudnn.benchmark = True
t.backends.cudnn.enabled = True
seed = 1
im_sz = 32
n_ch = 3
n_classes = 10
correct = 0
print = wlog

def hist_test_img_score(model, category, normal_test, abnormal_test):

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    model.eval()
    with t.no_grad():
        total_normal_score = []
        total_abnormal_score = []

        print('*** calculating normal img score.. ***')
        for normal_img in tqdm(normal_test):
            anomally_score = []
            slices = slice_tensor(normal_img, 4) # 불러오는 test img size 최적값을 parameter로서 찾아야됨!!!!!!!
            for slice in slices:
                slice = t.unsqueeze(slice, 0)
                slice = slice.to(device)
                anomally_score.append(model(slice).cpu())
            total_normal_score.append(max(anomally_score))
        
        print('*** calculating abnormal img score.. ***')
        for abnormal_img in tqdm(abnormal_test):
            anomally_score = []
            slices = slice_tensor(abnormal_img, 4) # 불러오는 test img size 최적값을 parameter로서 찾아야됨!!!!!!!
            for slice in slices:
                slice = t.unsqueeze(slice, 0)
                slice = slice.to(device)
                anomally_score.append(model(slice).cpu())
            total_abnormal_score.append(max(anomally_score))

    model.train()

    max = max(total_abnormal_score)
    min = min(total_normal_score)
    bins = np.linspace(min, max, len(total_abnormal_score)//1.2)

    plt.hist(total_normal_score, bins, alpha = 0.7, label = 'normal score')
    plt.hist(total_abnormal_score, bins, alpha = 0.7, label = 'abnormal score')
    plt.title(f'{category} score')
    plt.legend()


def slice_tensor(tensor, num_slices):
    sliced_tensors = []
    image_size = tensor.shape[2:]
    
    # 이미지 크기에서 4개로 자르기
    h_slices = torch.split(tensor, image_size[0] // num_slices, dim=2)
    for h_slice in h_slices:
        w_slices = torch.split(h_slice, image_size[1] // num_slices, dim=3)
        for w_slice in w_slices:
            sliced_tensors.append(w_slice)
    
    return sliced_tensors

def main(args):
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    f, _ = get_model_and_buffer(args, device)
    f.load_state_dict(t.load(f'{args.model_root}/last_ckpt.pt'))

    dload_test = get_data(args, im_shape=args.test_img_size, interpol=2, mode='test')

    category=['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 
              'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    
    transform_test = tr.Compose(
            [tr.ToTensor(),
             tr.Normalize((.5, .5, .5), (.5, .5, .5)),
             lambda x: x + args.sigma * t.randn_like(x)
            ])
    
    for cat in category:
        testset = MVTEC(args=args, root='mvtec', train=False, transform=transform_test,
                            resize=args.test_img_size, interpolation=2, category=[cat])

        testloader = t.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                                shuffle=False)

        dload_test = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

        test_img_list, test_label_list = [], []

        for test_img, label in dload_test:
            test_img_list.append(test_img)
            test_label_list.append(label)
        
        test_img_list, test_label_list = t.cat(test_img_list), t.cat(test_label_list)

        normal_test = test_img_list[test_label_list == 0]
        abnormal_test = test_img_list[test_label_list == 1]

        hist_test_img_score(f, cat, normal_test, abnormal_test)


    # dload_train, dload_test = get_data(args, im_shape=args.img_size, interpol=2)

    # normal_tmp_matrix=[]
    # abnormal_tmp_matrix=[]
    # normal_score_matrix=[]
    # abnormal_score_matrix=[]
    # for _ in range(args.num_slices**2):
    #     normal_tmp_matrix.append([])
    #     abnormal_tmp_matrix.append([])

    # for test_batch, label in dload_test:
    #     slices = slice_tensor(test_batch, args.num_slices)
    #     for i, slice in tqdm(enumerate(slices)):
    #         f, _ = get_model_and_buffer(args, device)
    #         f.load_state_dict(t.load(f'{args.model_root}/patch{i}_last_ckpt.pt'))
    #         print(f'*** patch{i}th model loaded ***')
    #         with t.no_grad():
    #             f.eval()
    #         for batch_num in range(len(test_batch)):
    #             score = f(slice[batch_num])
    #             score = score.logsumexp(1) # output이 10개로 되어있음. 임시방편
    #             if label[batch_num] == 0:
    #                 normal_tmp_matrix[i].append(score)
    #             else:
    #                 abnormal_tmp_matrix[i].append(score)
    
    # for j in range(len(normal_tmp_matrix[0])):
    #     normal_score_matrix.append([row[j] for row in normal_tmp_matrix])
    #     abnormal_score_matrix.append([row[j] for row in abnormal_tmp_matrix])
        
    # normal_max_values = [max(row) for row in normal_score_matrix]
    # abnormal_max_values = [max(row) for row in abnormal_score_matrix]
    
    # plt.hist((normal_max_values, abnormal_max_values), histtype='bar',label=('normal','abnormal'))
    # plt.xlabel('number')
    # plt.ylabel('count')
    # plt.title('histogram')
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("LDA Energy Based Models")
    parser.add_argument("--dataset", type=str, default="MVTec", choices=["MVTec", "cifar10", "svhn", "cifar100", 'tinyimagenet'])
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--model_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
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
    auto_select_gpu(args)
    init_debug(args)
    run_time = time.strftime('%m%d%H%M%S', time.localtime(time.time()))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    args.n_classes = 100 if "cifar100" in args.dataset else 10
    main(args)
    print(args.save_dir)
