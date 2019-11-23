from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import sys
import warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import models.imagenet as customized_models
from datasets.dataloader_imagenet_dct import valloader_upscaled_static
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from tensorboardX import SummaryWriter

default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-d', '--data', default='/mnt/ssd/kai.x', type=str)
parser.add_argument('--arch', '-a', metavar='ARCH', default='mobilenetv2dct',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--test-batch', default=200, type=int, metavar='N',
                    help='test batchsize (default: 200)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoints', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoints)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--cardinality', type=int, default=32, help='ResNeXt model cardinality (group).')
parser.add_argument('--base-width', type=int, default=4, help='ResNeXt model base width.')
parser.add_argument('--groups', type=int, default=3, help='ShuffleNet model groups')
# Miscs
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
# Device options
parser.add_argument('--gpu-id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--subset', default='0', type=str, help='subset of y, cb, cr')
args = parser.parse_args()

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_prec1 = 0

def main():
    global args, best_prec1

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    elif args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    baseWidth=args.base_width,
                    cardinality=args.cardinality,
                )
    elif args.arch.startswith('shufflenet'):
        model = models.__dict__[args.arch](
                    groups=args.groups
                )
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](channels=int(args.subset))

    criterion = nn.CrossEntropyLoss().cuda()

    # optionally resume from a checkpoint
    title = 'ImageNet-' + args.arch
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            args.checkpoint = os.path.dirname(args.resume)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    cudnn.benchmark = True

    val_loader = valloader_upscaled_static(args)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

def validate(val_loader, model, criterion):
    bar = Bar('Processing', max=len(val_loader))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (image, target) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            image, target = image.cuda(non_blocking=True), target.cuda(non_blocking=True)

            # compute output
            output = model(image)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))
            top5.update(prec5.item(), image.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=i + 1,
                        size=len(val_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
        bar.finish()
    return (losses.avg, top1.avg, top5.avg)

if __name__ == '__main__':
    main()
