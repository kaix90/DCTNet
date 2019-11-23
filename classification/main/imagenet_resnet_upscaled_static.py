from __future__ import print_function

import argparse
import os
import sys
import warnings
import shutil
import time
import random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed
import torch.optim as optim
import torch.utils.data as data
from datasets.dataset_imagenet_dct import ImageFolderDCT
import datasets.cvtransforms as transforms
from models.imagenet.resnet import ResNetDCT_Upscaled_Static
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from datasets.dataloader_imagenet_dct import valloader_upscaled_static
from tensorboardX import SummaryWriter

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('-d', '--data', default='path to dataset', type=str)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--test-batch', default=200, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('-c', '--checkpoint', default='checkpoints', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoints)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50dct',
                    help='model architecture: (default: resnet50dct)')
# Miscs
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--subset', default='192', type=str, help='subset of y, cb, cr')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--pretrained', default='True', type=str2bool,
                    help='load pretrained model or not')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_prec1 = 0  # best test accuracy

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


    model = ResNetDCT_Upscaled_Static(channels=int(args.subset), pretrained=args.pretrained)
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # Resume
    title = 'ImageNet-' + args.arch
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
        args.checkpoint = os.path.dirname(args.resume)

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model = model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # Data loading code
    val_loader = valloader_upscaled_static(args, model='resnet')

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc_top1, test_acc_top5 = test(val_loader, model, criterion)
        print(' Test Loss:  %.8f, Test Acc Top1:  %.2f, Test Acc Top5:  %.2f' % (test_loss, test_acc_top1, test_acc_top5))
        return

def test(val_loader, model, criterion):
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
        for batch_idx, (image, target) in enumerate(val_loader):
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
                        batch=batch_idx + 1,
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
