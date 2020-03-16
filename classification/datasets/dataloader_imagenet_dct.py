import os
import time
import torch
from datasets.dataset_imagenet_dct import ImageFolderDCT
import datasets.cvtransforms as transforms
from datasets import train_y_mean, train_y_std, train_cb_mean, train_cb_std, \
    train_cr_mean, train_cr_std
from datasets import train_y_mean_upscaled, train_y_std_upscaled, train_cb_mean_upscaled, train_cb_std_upscaled, \
    train_cr_mean_upscaled, train_cr_std_upscaled
from datasets import train_dct_subset_mean, train_dct_subset_std
from datasets import train_upscaled_static_mean, train_upscaled_static_std

def valloader_upscaled_static(args, model='mobilenet'):
    valdir = os.path.join(args.data, 'val')

    if model == 'mobilenet':
        input_size1 = 1024
        input_size2 = 896
    elif model == 'resnet':
        input_size1 = 512
        input_size2 = 448
    else:
        raise NotImplementedError
    if int(args.subset) == 0 or int(args.subset) == 192:
        transform = transforms.Compose([
                transforms.Resize(input_size1),
                transforms.CenterCrop(input_size2),
                transforms.Upscale(upscale_factor=2),
                transforms.TransformUpscaledDCT(),
                transforms.ToTensorDCT(),
                transforms.Aggregate(),
                transforms.NormalizeDCT(
                    train_upscaled_static_mean,
                    train_upscaled_static_std,
                )
            ])
    else:
        transform = transforms.Compose([
                transforms.Resize(input_size1),
                transforms.CenterCrop(input_size2),
                transforms.Upscale(upscale_factor=2),
                transforms.TransformUpscaledDCT(),
                transforms.ToTensorDCT(),
                transforms.SubsetDCT(channels=args.subset),
                transforms.Aggregate(),
                transforms.NormalizeDCT(
                    train_upscaled_static_mean,
                    train_upscaled_static_std,
                    channels=args.subset
                )
            ])

    val_loader = torch.utils.data.DataLoader(
        ImageFolderDCT(valdir, transform),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return val_loader

