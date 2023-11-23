import os
import torch
import torch as t
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch.nn.modules.loss import _Loss
from ExpUtils import AverageMeter
# data loader
from MVTec import MVTEC # 내 데이터로더 파일


class Hamiltonian(_Loss):

    def __init__(self, layer, reg_cof=1e-4):
        super(Hamiltonian, self).__init__()
        self.layer = layer
        self.reg_cof = 0

    def forward(self, x, p):

        y = self.layer(x)
        H = torch.sum(y * p)
        # H = H - self.reg_cof * l2
        return H


def sqrt(x):
    return int(t.sqrt(t.Tensor([x])))


def plot(p, x):
    return tv.utils.save_image(t.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def save_checkpoint(state, save, epoch):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, 'checkpt-%04d.pth' % epoch)
    torch.save(state, filename)


class DataSubset(Dataset):
    def __init__(self, base_dataset, inds=None, size=-1):
        self.base_dataset = base_dataset
        if inds is None:
            inds = np.random.choice(list(range(len(base_dataset))), size, replace=False)
        self.inds = inds

    def __getitem__(self, index):
        base_ind = self.inds[index]
        return self.base_dataset[base_ind]

    def __len__(self):
        return len(self.inds)


def cycle(loader):
    while True:
        for data in loader:
            yield data


def init_random(args, bs, im_sz=32, n_ch=3):
    return t.FloatTensor(bs, n_ch, im_sz, im_sz).uniform_(-1, 1)


def get_data(args, im_shape, interpol, mode, category, xy_std_dev):

    # *** 이 코드에서는 데이터셋을 불러올때부터 노이즈를 추가해주는 것을 보인다.(추측이여서 자세한 확인필요) ***
    if args.dataset == "MVTec":
        transform_train = tr.Compose(
            [#tr.Pad(4, padding_mode="reflect"),
             #tr.RandomCrop(im_sz),
             tr.ToTensor(),
             tr.Normalize((.5, .5, .5), (.5, .5, .5)),
             lambda x: x + args.sigma * t.randn_like(x)
             ]
        )

        transform_test = tr.Compose(
            [tr.ToTensor(),
             tr.Normalize((.5, .5, .5), (.5, .5, .5)),
             #lambda x: x + args.sigma * t.randn_like(x)
            ]
        )

    elif args.dataset == "MVTec_loco":
        pass
    # Data category to use: carpet, leather, wood, bottle, etc.

    if mode == 'train':

        trainset = MVTEC(args=args,root='mvtec', train=True, transform=transform_train, xy_std_dev=xy_std_dev,
                            resize=im_shape, interpolation=interpol, category=category)

        trainloader = t.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                shuffle=True)
        
        dload_train = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
        
        return dload_train
    
    elif mode == 'test':

        testset = MVTEC(args=args, root='mvtec', train=False, transform=transform_test,
                            resize=im_shape, interpolation=interpol, category=category)

        testloader = t.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                                shuffle=False)

        dload_test = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

        return dload_test


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def checkpoint(f, buffer, tag, args, device):
    f.cpu()
    ckpt_dict = {
        "model_state_dict": f.state_dict(), #f.module.state_dict()
        "replay_buffer": buffer,
    }
    t.save(ckpt_dict, os.path.join(args.save_dir, tag))
    f.to(device)


def set_bn_eval(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.eval()


def set_bn_train(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.train()


def eval_classification(f, dload, set_name, epoch, args=None, wlog=None):

    corrects, losses = [], []
    if args.n_classes >= 200:
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

    for x, y in dload:
        x, y = x.to(args.device), y.to(args.device)
        logits = f.classify(x)
        loss = nn.CrossEntropyLoss(reduction='none')(logits, y).detach().cpu().numpy()
        losses.extend(loss)
        if args.n_classes >= 200:
            acc1, acc5 = accuracy(logits, y, topk=(1, 5))
            top1.update(acc1[0], x.size(0))
            top5.update(acc5[0], x.size(0))
        else:
            correct = (logits.max(1)[1] == y).float().cpu().numpy()
            corrects.extend(correct)
        correct = (logits.max(1)[1] == y).float().cpu().numpy()
        corrects.extend(correct)
    loss = np.mean(losses)
    if wlog:
        my_print = wlog
    else:
        my_print = print
    if args.n_classes >= 200:
        correct = top1.avg
        my_print("Epoch %d, %s loss %.5f, top1 acc %.4f, top5 acc %.4f" % (epoch, set_name, loss, top1.avg, top5.avg))
    else:
        correct = np.mean(corrects)
        my_print("Epoch %d, %s loss %.5f, acc %.4f" % (epoch, set_name, loss, correct))
    if args.vis:

        args.writer.add_scalar('%s/Loss' % set_name, loss, epoch)
        if args.n_classes >= 200:
            args.writer.add_scalar('%s/Acc_1' % set_name, top1.avg, epoch)
            args.writer.add_scalar('%s/Acc_5' % set_name, top5.avg, epoch)
        else:
            args.writer.add_scalar('%s/Accuracy' % set_name, correct, epoch)
    return correct, loss
