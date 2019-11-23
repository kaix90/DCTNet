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


if __name__ == '__main__':
    import numpy as np
    from turbojpeg import TurboJPEG
    from jpeg2dct.numpy import load, loads

    jpeg_encoder = TurboJPEG('/usr/lib/libturbojpeg.so')


    # transform1 =transforms.Compose([
    #     transforms.RandomResizedCrop(224),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.TransformDCT(),  # 28x28x192
    #     transforms.DCTFlatten2D(),
    #     transforms.UpsampleDCT(upscale_ratio_h=4, upscale_ratio_w=4, debug=True),
    #     transforms.ToTensorDCT(),
    # ])

    # transform2 = transforms.Compose([
    #     transforms.DCTFlatten2D(),
    #     transforms.UpsampleDCT(upscale_ratio_h=4, upscale_ratio_w=4, debug=True),
    #     transforms.ToTensorDCT(),
    #     transforms.NormalizeDCT(
    #         train_y_mean_resized, train_y_std_resized,
    #         train_cb_mean_resized, train_cb_std_resized,
    #         train_cr_mean_resized, train_cr_std_resized),
    # ])

    # transform3 =transforms.Compose([
    #     transforms.RandomResizedCrop(224),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ResizedTransformDCT(),
    #     transforms.ToTensorDCT(),
    #     transforms.SubsetDCT(32),
    # ])

    transform4 = transforms.Compose([
        transforms.RandomResizedCrop(896),
        transforms.RandomHorizontalFlip(),
        transforms.Upscale(upscale_factor=2),
        transforms.TransformUpscaledDCT(),
        transforms.ToTensorDCT(),
        transforms.SubsetDCT(channels='24'),
        transforms.Aggregate(),
        transforms.NormalizeDCT(
            train_upscaled_static_mean,
            train_upscaled_static_std,
            channels='24'
        )
        ])

    transform5 = transforms.Compose([
        transforms.DCTFlatten2D(),
        transforms.UpsampleDCT(size_threshold=112 * 8, T=112 * 8, debug=False),
        transforms.SubsetDCT2(channels='32'),
        transforms.Aggregate2(),
        transforms.RandomResizedCropDCT(112),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensorDCT2(),
        transforms.NormalizeDCT(
            train_upscaled_static_mean,
            train_upscaled_static_std,
            channels='32'
        )
    ])
    # train_dataset = ImageFolderDCT('/ILSVRC2012/train', transform1, backend='opencv')
    # train_dataset = ImageFolderDCT('/ILSVRC2012/train', transform2
    # , backend='dct')
    train_dataset = ImageFolderDCT('/ILSVRC2012/train', transform5, backend='dct')

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16, shuffle=(train_sampler is None),
        num_workers=1, pin_memory=True, sampler=train_sampler)

    for i, data in enumerate(train_loader):
        print(data)


