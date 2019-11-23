# Optimized for DCT
# Upsampling in the compressed domain
import os
import sys
import random
from datasets.vision import VisionDataset
from PIL import Image
import cv2
import os.path
import numpy as np
import torch
from turbojpeg import TurboJPEG
from datasets import train_y_mean_resized, train_y_std_resized, train_cb_mean_resized, train_cb_std_resized, \
    train_cr_mean_resized, train_cr_std_resized
from jpeg2dct.numpy import loads

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):
    images = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def opencv_loader(path, colorSpace='YCrCb'):
    image = cv2.imread(str(path))
    # cv2.imwrite('/mnt/ssd/kai.x/work/code/iftc/datasets/cvtransforms/test/raw.jpg', image)
    if colorSpace == "YCrCb":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        # cv2.imwrite('/mnt/ssd/kai.x/work/code/iftc/datasets/cvtransforms/test/ycbcr.jpg', image)
    elif colorSpace == 'RGB':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def default_loader(path, backend='opencv', colorSpace='YCrCb'):
    from torchvision import get_image_backend
    if backend == 'opencv':
        return opencv_loader(path, colorSpace=colorSpace)
    elif get_image_backend() == 'accimage' and backend == 'acc':
        return accimage_loader(path)
    elif backend == 'pil':
        return pil_loader(path)
    else:
        raise NotImplementedError

def adjust_size(y_size, cbcr_size):
    if y_size == cbcr_size:
        return y_size, cbcr_size
    elif np.mod(y_size, 2) == 1:
        y_size -= 1
        cbcr_size = y_size // 2
    return y_size, cbcr_size

class DatasetFolderDCT(VisionDataset):
    def __init__(self, root, loader, extensions=None, transform=None, target_transform=None, is_valid_file=None,
                 backend='opencv', aggregate=False):
        super(DatasetFolderDCT, self).__init__(root)
        self.transform = transform
        self.target_transform = target_transform
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.backend=backend
        self.aggregate = aggregate

    def _find_classes(self, dir):
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        # path = '/storage-t1/user/kaixu/datasets/ILSVRC2012/test/train/n02447366/n02447366_23489.JPEG'
        # print('{}, {}'.format(index, path))
        if self.backend == 'opencv':
            sample = self.loader(path, backend='opencv', colorSpace='BGR')
        elif self.backend == 'dct':
            try:
                with open(path, 'rb') as src:
                    buffer = src.read()
                dct_y, dct_cb, dct_cr = loads(buffer)
            except:
                notValid = True
                while notValid:
                    index = random.randint(0, len(self.samples)-1)
                    path, target = self.samples[index]
                    with open(path, 'rb') as src:
                        buffer = src.read()
                    try:
                        dct_y, dct_cb, dct_cr = loads(buffer)
                        notValid = False
                    except:
                        notValid = True

            if len(dct_y.shape) != 3:
                notValid = True
                while notValid:
                    index = random.randint(0, len(self.samples)-1)
                    path, target = self.samples[index]
                    with open(path, 'rb') as src:
                        buffer = src.read()
                    try:
                        dct_y, dct_cb, dct_cr = loads(buffer)
                        notValid = False
                    except:
                        print(path)
                        notValid = True

            y_size_h, y_size_w = dct_y.shape[:-1]
            cbcr_size_h, cbcr_size_w = dct_cb.shape[:-1]

            y_size_h, cbcr_size_h = adjust_size(y_size_h, cbcr_size_h)
            y_size_w, cbcr_size_w = adjust_size(y_size_w, cbcr_size_w)
            dct_y = dct_y[:y_size_h, :y_size_w]
            dct_cb = dct_cb[:cbcr_size_h, :cbcr_size_w]
            dct_cr = dct_cr[:cbcr_size_h, :cbcr_size_w]
            sample = [dct_y, dct_cb, dct_cr]

            y_h, y_w, _ = dct_y.shape
            cbcr_h, cbcr_w, _ = dct_cb.shape

        if self.transform is not None:
            dct_y, dct_cb, dct_cr = self.transform(sample)

        if self.backend == 'dct':
            if dct_cb is not None:
                image = torch.cat((dct_y, dct_cb, dct_cr), dim=1)
                return image, target
            else:
                return dct_y, target
        else:
            if dct_cb is not None:
                return dct_y, dct_cb, dct_cr, target
            else:
                return dct_y, target

    def __len__(self):
        return len(self.samples)

class ImageFolderDCT(DatasetFolderDCT):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None, backend='opencv', aggregate=False):
        super(ImageFolderDCT, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file,
                                          backend=backend,
                                          aggregate=aggregate)
        self.imgs = self.samples


if __name__ == '__main__':
    dataset = 'imagenet'

    import torch
    import datasets.cvtransforms as transforms
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import minmax_scale

    jpeg_encoder = TurboJPEG('/home/kai.x/work/local/lib/libturbojpeg.so')
    # jpeg_encoder = TurboJPEG('/usr/lib/libturbojpeg.so')
    if dataset == 'imagenet':
        input_normalize = []
        input_normalize_y = transforms.Normalize(mean=train_y_mean_resized,
                                               std=train_y_std_resized)
        input_normalize_cb = transforms.Normalize(mean=train_cb_mean_resized,
                                                std=train_cb_std_resized)
        input_normalize_cr = transforms.Normalize(mean=train_cr_mean_resized,
                                                std=train_cr_std_resized)
        input_normalize.append(input_normalize_y)
        input_normalize.append(input_normalize_cb)
        input_normalize.append(input_normalize_cr)
        val_loader = torch.utils.data.DataLoader(
            # ImageFolderDCT('/mnt/ssd/kai.x/dataset/ILSVRC2012/val', transforms.Compose([
            ImageFolderDCT('/storage-t1/user/kaixu/datasets/ILSVRC2012/val', transforms.Compose([
                transforms.ToYCrCb(),
                transforms.TransformDCT(),
                transforms.UpsampleDCT(T=896, debug=False),
                transforms.CenterCropDCT(112),
                transforms.ToTensorDCT(),
                transforms.NormalizeDCT(
                    train_y_mean_resized, train_y_std_resized,
                    train_cb_mean_resized, train_cb_std_resized,
                    train_cr_mean_resized, train_cr_std_resized),
            ])),
            batch_size=1, shuffle=False,
            num_workers=1, pin_memory=False)

        # train_dataset = ImageFolderDCT('/mnt/ssd/kai.x/dataset/ILSVRC2012/train', transforms.Compose([
        train_dataset = ImageFolderDCT('/storage-t1/user/kaixu/datasets/ILSVRC2012/train', transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToYCrCb(),
            transforms.ChromaSubsample(),
            transforms.UpsampleDCT(size=224, T=896, debug=False),
            transforms.ToTensorDCT(),
            transforms.NormalizeDCT(
                train_y_mean_resized,  train_y_std_resized,
                train_cb_mean_resized, train_cb_std_resized,
                train_cr_mean_resized, train_cr_std_resized),
        ]))

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1, shuffle=False,
            num_workers=1, pin_memory=False)

    from torchvision.utils import save_image
    dct_y_mean_total, dct_y_std_total = [], []
    # for batch_idx, (dct_y, dct_cb, dct_cr, targets) in enumerate(val_loader):
    for batch_idx, (dct_y, dct_cb, dct_cr, targets) in enumerate(train_loader):
        coef = dct_y.numpy()
        dct_y_mean, dct_y_std = [], []

        for c in coef:
            c = c.reshape((64, -1))
            dct_y_mean.append([np.mean(x) for x in c])
            dct_y_std.append([np.std(x) for x in c])

        dct_y_mean_np = np.asarray(dct_y_mean).mean(axis=0)
        dct_y_std_np = np.asarray(dct_y_std).mean(axis=0)
        dct_y_mean_total.append(dct_y_mean_np)
        dct_y_std_total.append(dct_y_std_np)
        # print('The mean of dct_y is: {}'.format(dct_y_mean_np))
        # print('The std of dct_y is: {}'.format(dct_y_std_np))

    print('The mean of dct_y is: {}'.format(np.asarray(dct_y_mean_total).mean(axis=0)))
    print('The std of dct_y is: {}'.format(np.asarray(dct_y_std_total).mean(axis=0)))


