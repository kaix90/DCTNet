'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import errno
import os
import sys
import time
import math
import cv2
import torchvision

import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch

__all__ = ['get_mean_and_std_yuv', 'init_params', 'mkdir_p', 'AverageMeter']
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# from turbojpeg import TurboJPEG
from datasets.dataset_imagenet_dct import ImageFolderDCT
import datasets.cvtransforms as transforms


def get_mean_and_std_dct_resized(dataset, model='mobilenet'):
    '''Compute the mean and std value of dataset.'''

    # Dataset = ImageFolderDCT(dataset, transforms.Compose([
    #     transforms.TransformDCT(),  # 28x28x192
    #     transforms.DCTFlatten2D(),
    #     transforms.UpsampleDCT(upscale_ratio_h=4, upscale_ratio_w=4, debug=False),
    #     transforms.ToTensorDCT(),
    #     transforms.Average()
    # ]), aggregate=True)

    # Dataset = ImageFolderDCT(dataset, transforms.Compose([
    #     transforms.Upscale(upscale_factor=2),
    #     transforms.TransformUpscaledDCT(),
    #     transforms.ToTensorDCT(),
    #     transforms.Aggregate(),
    #     transforms.Average()
    # ]))
    if model == 'mobilenet':
        input_size = 896
        batchsize = 256
    elif model == 'resnet':
        input_size = 448
        batchsize = 128
    else:
        raise NotImplementedError

    Dataset = ImageFolderDCT(dataset, transforms.Compose([
        transforms.DCTFlatten2D(),
        transforms.UpsampleDCT(size_threshold=input_size, T=input_size, debug=False),
        # transforms.UpsampleDCT(size_threshold=112 * 8, T=112 * 8, debug=False),
        transforms.Aggregate2(),
        # transforms.RandomResizedCropDCT(112),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensorDCT2(),

        transforms.Average()
    ]), backend='dct')

    dataloader = torch.utils.data.DataLoader(Dataset, batch_size=batchsize, pin_memory=True, shuffle=False, num_workers=16)

    mean, std = torch.zeros(192), torch.zeros(192)
    print('==> Computing mean and std..')

    # end = time.time()
    for i, (inputs, targets) in enumerate(dataloader):
        # print('data time: {}'.format(time.time()-end))
        print('{}/{}'.format(i, len(dataloader)))

        mean += inputs.mean(dim=0)
        std += inputs.std(dim=0)
        # end = time.time()

    mean.div_(i+1)
    std.div_(i+1)

    return mean, std

def get_mean_and_std_dct(dataset, sublist=None):
    '''Compute the mean and std value of dataset.'''
    import datasets.cvtransforms as transforms
    # jpeg_encoder = TurboJPEG('/home/kai.x/work/local/lib/libturbojpeg.so')

    # Dataset = ImageFolderDCT(dataset, transforms.Compose([
    #     #     transforms.RandomResizedCrop(224),
    #     #     # transforms.RandomHorizontalFlip(),
    #     #     transforms.TransformDCT(),
    #     #     transforms.UpsampleDCT(896),
    #     #     transforms.ToTensorDCT()
    #     #         # transforms.RandomResizedCrop(256),
    #     #         # transforms.RandomHorizontalFlip(),
    #     #         # transforms.TransformDCT(),
    #     #         # transforms.UpsampleDCT(256, 256),
    #     #         # transforms.CenterCropDCT(112),
    #     #         # transforms.ToTensorDCT()
    #     #     ]))

    # Dataset = ImageFolderDCT(dataset, transforms.Compose([
    #     transforms.Upscale(),
    #     transforms.TransformDCT(),
    #     transforms.ToTensorDCT(),
    # ]))

    Dataset = ImageFolderDCT(dataset, transforms.Compose([
        transforms.DCTFlatten2D(),
        transforms.UpsampleDCT(upscale_ratio_h=4, upscale_ratio_w=4, debug=False),
        transforms.ToTensorDCT(),
        transforms.Average()
    ]), backend='dct')

    dataloader = torch.utils.data.DataLoader(Dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    mean_y, mean_cb, mean_cr = torch.zeros(64), torch.zeros(64), torch.zeros(64)
    std_y, std_cb, std_cr = torch.zeros(64), torch.zeros(64), torch.zeros(64)
    print('==> Computing mean and std..')

    end = time.time()
    if sublist is None:
        for i, (inputs_y, inputs_cb, inputs_cr, targets) in enumerate(dataloader):
            print('data time: {}'.format(time.time()-end))
            print('{}/{}'.format(i, len(dataloader)))

            mean_y += inputs_y.mean(dim=0)
            std_y += inputs_y.std(dim=0)
            mean_cb += inputs_cb.mean(dim=0)
            std_cb += inputs_cb.std(dim=0)
            mean_cr += inputs_cr.mean(dim=0)
            std_cr += inputs_cr.std(dim=0)
            end = time.time()

        mean_y.div_(i+1)
        std_y.div_(i+1)
        mean_cb.div_(i+1)
        std_cb.div_(i+1)
        mean_cr.div_(i+1)
        std_cr.div_(i+1)
    else:
        dataloader_iterator = iter(dataloader)
        for i in range(sublist):
            try:
                inputs_y, inputs_cb, inputs_cr, targets = next(dataloader_iterator)
            except:
                print('error')

            print('{}/{}'.format(i, sublist))
            for i in range(64):
                mean_y[i] += inputs_y[:, i, :, :].mean()
                std_y[i] += inputs_y[:, i, :, :].std()
                mean_cb[i] += inputs_cb[:, i, :, :].mean()
                std_cb[i] += inputs_cb[:, i, :, :].std()
                mean_cr[i] += inputs_cr[:, i, :, :].mean()
                std_cr[i] += inputs_cr[:, i, :, :].std()
        mean_y.div_(sublist)
        std_y.div_(sublist)
        mean_cb.div_(sublist)
        std_cb.div_(sublist)
        mean_cr.div_(sublist)
        std_cr.div_(sublist)
    return mean_y, std_y, mean_cb, std_cb, mean_cr, std_cr

def yuv_loader(path):
    image = cv2.imread(str(path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    return image

def get_mean_and_std_yuv(dataset):
    '''Compute the mean and std value of dataset.'''

    dataloader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(dataset, transforms.Compose([
            transforms.RandomResizedCropDCT(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.AverageYUV()
        ]), loader=yuv_loader),
        batch_size=128, shuffle=False,
        num_workers=16)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for idx, (inputs, targets) in enumerate(dataloader):
        mean += inputs.mean(dim=0)
        std += inputs.std(dim=0)
        # for i in range(3):
        #     mean[i] += inputs[:,i].mean()
        #     std[i] += inputs[:,i].std()
    mean.div_(idx+1)
    std.div_(idx+1)
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    # dataset = '/mnt/ssd/kai.x/val'
    dataset = '/ILSVRC2012/val'
    # dataset = '/mnt/ssd/kai.x/train'
    #print(get_mean_and_std_dct(dataset, sublist=200000))
    # print(get_mean_and_std_yuv(dataset))
    print(get_mean_and_std_dct_resized(dataset, model='resnet'))
    # print(get_mean_and_std_dct_resized(dataset, model='mobilenet'))

