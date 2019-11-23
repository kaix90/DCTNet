import cv2
import mmcv
import numpy as np
from imagecorruptions import corrupt
from numpy import random

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from ..registry import PIPELINES
from turbojpeg import TurboJPEG
from jpeg2dct.numpy import load, loads
from mmdet.datasets.pipelines.dct_channel_index import dct_channel_index
from mmdet.utils.plot_dct import plot_dct

INTER_MODE = {'NEAREST': cv2.INTER_NEAREST, 'BILINEAR': cv2.INTER_LINEAR, 'BICUBIC': cv2.INTER_CUBIC}
PAD_MOD = {'constant': cv2.BORDER_CONSTANT,
           'edge': cv2.BORDER_REPLICATE,
           'reflect': cv2.BORDER_DEFAULT,
           'symmetric': cv2.BORDER_REFLECT
           }

@PIPELINES.register_module
class ToDCT(object):
    def __init__(self):
        self.jpeg = TurboJPEG('/usr/lib/libturbojpeg.so')

    def __call__(self, results):
        img = np.ascontiguousarray(results['img'], dtype="uint8")
        img_encode = self.jpeg.encode(img, quality=100, jpeg_subsample=2)
        dct_y, dct_cb, dct_cr = loads(img_encode)   # 28
        results['dct_y'] = dct_y
        results['dct_cb'] = dct_cb
        results['dct_cr'] = dct_cr
        return results

@PIPELINES.register_module
class ToDCTUpscaledStatic(object):
    def __init__(self, channels=None, is_test=False, interpolation='BILINEAR'):
        self.jpeg = TurboJPEG('/usr/lib/libturbojpeg.so')
        self.channels = channels
        self.is_test = is_test
        self.interpolation = interpolation

        if channels and channels != 192:
            self.subset_channel_index = dct_channel_index
            self.subset_y = self.subset_channel_index[channels][0]
            self.subset_cb = self.subset_channel_index[channels][1]
            self.subset_cr = self.subset_channel_index[channels][2]

    def __call__(self, results):
        h, w = results['img'].shape[:-1]
        if self.is_test:
            results['img_raw'] = results['img']
        img_raw_4x = cv2.resize(results['img'], dsize=(w*2, h*2), interpolation=INTER_MODE[self.interpolation])
        img_raw_8x = cv2.resize(results['img'], dsize=(w*4, h*4), interpolation=INTER_MODE[self.interpolation])
        img_4x = np.ascontiguousarray(img_raw_4x, dtype="uint8")
        img_8x = np.ascontiguousarray(img_raw_8x, dtype="uint8")
        img_encode_4x = self.jpeg.encode(img_4x, quality=100, jpeg_subsample=2)
        img_encode_8x = self.jpeg.encode(img_8x, quality=100, jpeg_subsample=2)
        dct_y, _, _ = loads(img_encode_4x)   # 28
        _, dct_cb, dct_cr = loads(img_encode_8x)   # 28


        plot_dct(dct_y, results['img_info']['filename'])

        if self.channels == 192:
            results['img'] = np.concatenate((dct_y, dct_cb, dct_cr), axis=2)
        else:
            results['img'] = np.concatenate((dct_y[:, :, self.subset_y], dct_cb[:, :, self.subset_cb],
                                             dct_cr[:, :, self.subset_cr]), axis=2)


        return results
