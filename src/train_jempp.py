# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required blicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch as t
import torch.nn as nn
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from ExpUtils import *
from utils import eval_classification, Hamiltonian, checkpoint, get_data, set_bn_train, set_bn_eval, plot
from models.jem_models import get_model_and_buffer
from torch.utils.tensorboard.writer import SummaryWriter
from copy import deepcopy
import random
import matplotlib.pyplot as pl
import torchvision.transforms as tr
from torchvision import transforms
from sklearn.metrics import roc_auc_score
t.set_num_threads(2)
t.backends.cudnn.benchmark = True
t.backends.cudnn.enabled = True
seed = 1
inner_his = []
conditionals = []


def init_random(args, bs):
    global conditionals
    n_ch = 3
    size = [3, args.img_size, args.img_size]
    im_sz = args.img_size
    new = t.zeros(bs, n_ch, im_sz, im_sz)
    for i in range(bs):
        dist = conditionals[0]
        new[i] = dist.sample().view(size)
    return t.clamp(new, -1, 1).cpu()


def sample_p_0(replay_buffer, bs, y=None):
    if len(replay_buffer) == 0:
        return init_random(args, bs), []
    buffer_size = len(replay_buffer) # if y is None else len(replay_buffer) // args.n_classes
    inds = t.randint(0, buffer_size, (bs,))
    # if cond, convert inds to class conditional inds
    # if y is not None:
    #     inds = y.cpu() * buffer_size + inds
    buffer_samples = replay_buffer[inds]
    random_samples = init_random(args, bs)
    choose_random = (t.rand(bs) < args.reinit_freq).float()[:, None, None, None]
    samples = choose_random * random_samples + (1 - choose_random) * buffer_samples
    return samples.to(args.device), inds


def sample_q(f, replay_buffer, y=None, n_steps=10, in_steps=10, args=None, save=True):
    """this func takes in replay_buffer now so we have the option to sample from
    scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
    """
    global inner_his
    inner_his = []
    # Batch norm uses train status
    # f.eval()
    # get batch size
    bs = args.batch_size #if y is None else y.size(0)
    # generate initial samples and buffer inds of those samples (if buffer is used)
    init_sample, buffer_inds = sample_p_0(replay_buffer, bs=bs, y=y)
    x_k = t.autograd.Variable(init_sample, requires_grad=True)
    # sgld
    if in_steps > 0:
        Hamiltonian_func = Hamiltonian(f.f.layer_one)

    eps = args.eps
    if args.pyld_lr <= 0:
        in_steps = 0

    for it in range(n_steps):
        energies = f(x_k, y=y)
        e_x = energies.sum()
        # wgrad = f.f.conv1.weight.grad
        eta = t.autograd.grad(e_x, [x_k], retain_graph=True)[0]
        # e_x.backward(retain_graph=True)
        # eta = x_k.grad.detach()
        # f.f.conv1.weight.grad = wgrad

        if in_steps > 0:
            p = 1.0 * f.f.layer_one_out.grad
            p = p.detach()

        tmp_inp = x_k.data
        tmp_inp.requires_grad_()
        if args.sgld_lr > 0:
            # if in_steps == 0: use SGLD other than PYLD
            # if in_steps != 0: combine outter and inner gradients
            # default 0
            tmp_inp = x_k + t.clamp(eta, -eps, eps) * args.sgld_lr
            tmp_inp = t.clamp(tmp_inp, -1, 1)

        for i in range(in_steps):

            H = Hamiltonian_func(tmp_inp, p)

            eta_grad = t.autograd.grad(H, [tmp_inp], only_inputs=True, retain_graph=True)[0]
            eta_step = t.clamp(eta_grad, -eps, eps) * args.pyld_lr

            tmp_inp.data = tmp_inp.data + eta_step
            tmp_inp = t.clamp(tmp_inp, -1, 1)

        x_k.data = tmp_inp.data

        if args.sgld_std > 0.0:
            x_k.data += args.sgld_std * t.randn_like(x_k)

    if in_steps > 0:
        loss = -1.0 * Hamiltonian_func(x_k.data, p)
        loss.backward()

    f.train()
    final_samples = x_k.detach()
    # update replay buffer
    if len(replay_buffer) > 0 and save:
        replay_buffer[buffer_inds] = final_samples.cpu()
    return final_samples

def category_mean(dload_train, args):
    import time
    start = time.time()
    size = [3, args.img_size, args.img_size]

    centers = t.zeros([1, int(np.prod(size))])
    covs = t.zeros([1, int(np.prod(size)), int(np.prod(size))])

    im_test, targ_test = [], []
    for _ in range(args.n_randcrop):
        for im, targ in dload_train:
            im_test.append(im)
            targ_test.append(targ)
        
    im_test, targ_test = t.cat(im_test), t.cat(targ_test)

    imc = im_test
    imc = imc.view(len(imc), -1)
    mean = imc.mean(dim=0)
    sub = imc - mean.unsqueeze(dim=0)
    cov = sub.t() @ sub / len(imc)
    centers[0] = mean
    covs[0] = cov
    print(time.time() - start)
    t.save(centers, '%s/%s_mean.pt' % (args.output_root, args.dataset))
    t.save(covs, '%s/%s_cov.pt' % (args.output_root, args.dataset))

# *** 추정된 mean과 cov를 통해 분포 생성 ***
def init_from_centers(args):
    global conditionals
    from torch.distributions.multivariate_normal import MultivariateNormal
    bs = args.buffer_size
    size = [3, args.img_size, args.img_size]
    print('init_from_centers : load mean and cov')
    centers = t.load('%s/%s_mean.pt' % (args.output_root, args.dataset))
    covs = t.load('%s/%s_cov.pt' % (args.output_root, args.dataset))
    print(f'init_from_centers : len(mean):{len(centers)}, len(cov):{len(covs)}')

    buffer = []

    mean = centers[0].to(args.device)
    cov = covs[0].to(args.device)
    dist = MultivariateNormal(mean, covariance_matrix=cov + 1e-4 * t.eye(int(np.prod(size))).to(args.device))
    buffer.append(dist.sample((bs, )).view([bs] + size).cpu())
    conditionals.append(dist)
    
    return t.clamp(t.cat(buffer), -1, 1)

def slice_tensor(tensor, num_slices):
    sliced_tensors = []
    image_size = tensor.shape[1:]
    
    # 이미지 크기에서 4개로 자르기
    h_slices = t.split(tensor, image_size[0] // num_slices, dim=1)
    for h_slice in h_slices:
        w_slices = t.split(h_slice, image_size[1] // num_slices, dim=2)
        for w_slice in w_slices:
            sliced_tensors.append(w_slice)
    
    return sliced_tensors

def get_anomaly_score(args, model, test_img_list):

    model.eval()
    with t.no_grad():
        total_energy_score = []

        for test_img in test_img_list:
            anomally_score = []
            slices = slice_tensor(test_img, args.n_slices) # 불러오는 test img size 최적값을 parameter로서 찾아야됨!!!!!!!
            for slice in slices:
                slice = t.unsqueeze(slice, 0)
                slice = slice.to(args.device)
                anomally_score.append(model(slice).cpu())
            total_energy_score.append(max(anomally_score))

    total_energy_score = t.tensor(total_energy_score)
    model.train()

    return total_energy_score

def plt_score(args, category, total_energy_score, test_label_list, epoch, auroc, threshold):

    total_energy_score = minmax_scale(total_energy_score)
    total_abnormal_score = total_energy_score[test_label_list==0]
    total_normal_score = total_energy_score[test_label_list==1]
    bins = np.linspace(0, 1,len(total_energy_score)+10)

    plt.figure(figsize=(9,5), dpi=200)
    plt.hist(total_normal_score, bins, alpha=0.7, label='normal score')
    plt.hist(total_abnormal_score, bins, alpha=0.7, label='abnormal score')
    plt.axvline(threshold, color='red', linestyle='dashed', linewidth=0.5, label=f'Threshold = {round(threshold,3)}')
    plt.title(f'{category} score')
    plt.legend()
    plt.savefig(f'{args.save_dir}/result_img/({args.category})_epoch_{str(epoch).zfill(3)}_auroc_{round(auroc,3)}_result.png', dpi=200)
    plt.cla()   # clear the current axes
    plt.clf()   # clear the current figure
    plt.close() # closes the current figure


def calc_auroc(args, total_energy_score, test_label_list):

    best_acc = 0
    best_auroc = 0
    best_threshold_auroc = None

    # for normal score
    for i in total_energy_score[test_label_list==1]:
        threshold = float(i)
        pred = list(map(int, total_energy_score <= threshold))
        label = test_label_list
        try:
            auroc = roc_auc_score(label, pred)
        except ValueError:
            auroc = 0

        if auroc > best_auroc:
            best_auroc = auroc
            best_threshold_auroc = threshold

        label, pred = np.array(label), np.array(pred)
        acc = sum(label == pred)/len(label) * 100

        if acc > best_acc:
            best_acc = acc

    return best_auroc, best_acc, best_threshold_auroc

def minmax_scale(anomaly_score):
    min_value = anomaly_score.min()
    max_value = anomaly_score.max()
    anomaly_score = (anomaly_score - min_value) / (max_value - min_value)
    return anomaly_score


def main(args):
    global conditionals
    random.seed(args.seed)
    np.random.seed(args.seed)
    t.manual_seed(args.seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(args.seed)

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    args.device = device

    summary_log_dir = os.path.join(args.output_root, 'log')
    summary_writer = SummaryWriter(log_dir=summary_log_dir)
    output_dir = args.save_dir + '/result_img/'
    os.makedirs(output_dir)
    
    category = ['carpet']#['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    xy_std_dev_list = [(300, 300)]#[(300, 300), (140, 140), (260, 50), (300, 300), (300, 300), (170, 170), (300, 300), (130, 130), (150, 80), (300, 300), (300, 300), (70, 300), (80, 200), (300, 300), (300, 300)]

    for cat, xy_std_dev in tqdm(zip(category, xy_std_dev_list)):
        #for dataset centers
        #if not os.path.isfile('%s_cov.pt' % args.dataset):
        print(f'calculate category_mean! : {cat}')
        dload_train = get_data(args, im_shape=args.img_size, interpol=2, mode='train', xy_std_dev=xy_std_dev, category=[cat])
        category_mean(dload_train, args)

        print(f'Category : {cat} Train start!')
        f, replay_buffer = get_model_and_buffer(args, device)
        replay_buffer = init_from_centers(args)
        dload_test = get_data(args, im_shape=args.test_img_size, interpol=2, mode='test', xy_std_dev=None, category=[cat])
        print(f'Conditionals check : {len(conditionals)}')

        # for validation
        test_img_list, test_label_list = [], []

        for test_img, label in dload_test:
            test_img_list.append(test_img)
            test_label_list.append(label)
        
        test_img_list, test_label_list = t.cat(test_img_list), t.cat(test_label_list)

        # optimizer
        params = f.parameters()
        if args.optimizer == "adam":
            optim = t.optim.Adam(params, lr=args.lr, betas=[.9, .999], weight_decay=args.weight_decay)
        else:
            optim = t.optim.SGD(params, lr=args.lr, momentum=.9, weight_decay=args.weight_decay)

        cur_iter = 1
        n_steps = args.n_steps
        in_steps = args.in_steps

        best_auroc = 0.0
        best_model = None
        best_sample = None
        best_epoch = None
        best_threshold = None

        # train
        for epoch in tqdm(range(1, args.n_epochs+1)):

            total_L = []

            for j, (x_p_d, label) in enumerate(dload_train):

                x_p_d = x_p_d.to(device)

                L = 0.

                x_q = sample_q(f, replay_buffer, y=None, n_steps=n_steps, in_steps=in_steps, args=args)

                # *** 가짜 샘플 진짜 샘플로 loss 구하기 ***
                fp_all = f(x_p_d)
                fq_all = f(x_q)
                fp = fp_all.mean()
                fq = fq_all.mean()

                l_p_x = -(fp - fq)

                #reg_L = args.alpha * (fp ** 2 + fq ** 2).mean()
                L += l_p_x #+ reg_L
                total_L.append(int(L))

                if L.abs().item() > 1e8:
                    print("BAD BOIIIIIIIIII")
                    print("min {:>4.3f} max {:>5.3f}".format(x_q.min().item(), x_q.max().item()))
                    plot('{}/diverge_{}_{:>06d}.png'.format(args.save_dir, epoch, j), x_q)
                    return

                optim.zero_grad()
                L.backward()
                optim.step()
                cur_iter += 1

            total_L = np.mean(total_L)
            

            # validation
            total_energy_score = get_anomaly_score(args, f, test_img_list)
            auroc, acc, threshold = calc_auroc(args, total_energy_score, test_label_list)
            print('\nepoch : {} -> P(x) | Loss={:>9.4f} test auroc = {:>9.4f} test acc = {:>9.4f} threshold = {:>9.4f}'.format(epoch, total_L, auroc, acc, threshold))

            summary_writer.add_scalar(f'({cat}) Loss', total_L, global_step=epoch)
            summary_writer.add_scalar(f'({cat}) test accuracy', acc, global_step=epoch)
            summary_writer.add_scalar(f'({cat}) test auroc', auroc, global_step=epoch)
            summary_writer.add_scalar(f'({cat}) threshold', threshold, global_step=epoch)

            if auroc > best_auroc:
                best_auroc = auroc
                best_model = deepcopy(f)
                best_sample = sample_q(f, replay_buffer, n_steps=n_steps, in_steps=in_steps, args=args)
                best_epoch = epoch
                best_threshold = threshold
                checkpoint(f, replay_buffer, f'({cat})best_model_ckpt_auroc-{round(auroc,3)}_epoch-{str(epoch).zfill(3)}_theshold-{round(threshold,3)}.pt', args, device)
                plt_score(args, cat, total_energy_score, test_label_list, epoch, auroc, threshold)

            if epoch % args.print_every == 0:
                x_q = sample_q(f, replay_buffer, n_steps=n_steps, in_steps=in_steps, args=args)
                plot('{}/samples/({})x_q_epoch-{}.png'.format(args.save_dir, cat, str(epoch).zfill(3)), x_q)

            if best_auroc == 1.0:
                break

        #checkpoint(best_model, replay_buffer, f'({cat})best_model_ckpt_auroc-{round(best_auroc,3)}_epoch-{str(best_epoch).zfill(3)}_theshold-{round(best_threshold,3)}.pt', args, device)
        plot('{}/samples/({})best_model_sample_auroc-{}_epoch-{}.png'.format(args.save_dir, cat, round(best_auroc,3), best_epoch), best_sample)
        conditionals = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser("LDA Energy Based Models")
    parser.add_argument("--dataset", type=str, default="MVTec", choices=["MVTec", "cifar10", "svhn", "cifar100", 'tinyimagenet'])
    parser.add_argument("--data_root", type=str, default='/app/input/dataset/mvtec')
    parser.add_argument("--output_root", type=str, default='/app/outputs')
    parser.add_argument("--saved_pic_root", type=str, default='/app/outputs/result_img')

    # Data Loader
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--test_img_size", type=int, default=896)
    parser.add_argument("--n_slices", type=int, default=7)
    # parser.add_argument("--x_std_dev", type=int, default=300)
    # parser.add_argument("--y_std_dev", type=int, default=300)
    parser.add_argument("--n_classes", type=int, default=15) #원래 15
    parser.add_argument("--category", type=str, default=None, choices=['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper'])
    

    # Random Crop
    parser.add_argument("--n_randcrop", type=int, default=25) # 한개의 이미지당 random crop해서 가져올 개수
    parser.add_argument("--crop_size", type=int, default=256) # crop size
    
    # optimization
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay_epochs", nargs="+", type=int, default=[60, 90, 120, 135], help="decay learning rate by decay_rate at these epochs")
    parser.add_argument("--decay_rate", type=float, default=.2, help="learning rate decay multiplier")
    parser.add_argument("--clf_only", action="store_true", default=False, help="If set, then only train the classifier")
    # parser.add_argument("--labels_per_class", type=int, default=-1,
    #                     help="number of labeled examples per class, if zero then use all labels")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_epochs", type=int, default=200)
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
    parser.add_argument("--n_steps", type=int, default=20, help="number of steps of SGLD per iteration, 20 works for PCD")
    parser.add_argument("--in_steps", type=int, default=10, help="number of steps of SGLD per iteration, 100 works for short-run, 20 works for PCD")
    parser.add_argument("--width", type=int, default=10, help="WRN width parameter")
    parser.add_argument("--depth", type=int, default=28, help="WRN depth parameter")
    parser.add_argument("--uncond", action="store_true", default = False, help="If set, then the EBM is unconditional")
    parser.add_argument("--class_cond_p_x_sample", action="store_true", default = True,
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
    parser.add_argument("--ckpt_every", type=int, default=1, help="Epochs between checkpoint save")
    parser.add_argument("--eval_every", type=int, default=1, help="Epochs between evaluation")
    parser.add_argument("--print_every", type=int, default=50, help="Iterations between print")
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
    # parser.add_argument("--gpu-id", type=str, default="0")
    args = parser.parse_args()
    init_env(args, logger)
    args.save_dir = args.dir_path
    os.makedirs('{}/samples'.format(args.dir_path))
    print = wlog
    print(args.dir_path)
    main(args)
    print(args.dir_path)

