from collections.abc import Sequence

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines.dct_channel_index import dct_channel_index
from .formating import to_tensor

from ..registry import PIPELINES

@PIPELINES.register_module
class DefaultFormatBundleDCT(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """

    def __call__(self, results):
        if 'img' in results:
            dct_y = np.ascontiguousarray(results['dct_y'].transpose(2, 0, 1))
            dct_cb = np.ascontiguousarray(results['dct_cb'].transpose(2, 0, 1))
            dct_cr = np.ascontiguousarray(results['dct_cr'].transpose(2, 0, 1))
            results['dct_y'] = DC(to_tensor(dct_y), stack=True)
            results['dct_cb'] = DC(to_tensor(dct_cb), stack=True)
            results['dct_cr'] = DC(to_tensor(dct_cr), stack=True)
        for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))
        if 'gt_masks' in results:
            results['gt_masks'] = DC(results['gt_masks'], cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]), stack=True)
        return results

    def __repr__(self):
        return self.__class__.__name__

@PIPELINES.register_module
class NormalizeDCT(object):
    """Normalize the image.
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True, to_rgb_raw=False, inputnorm=False):
        self.mean_y = np.array(mean[0], dtype=np.float32)
        self.std_y = np.array(std[0], dtype=np.float32)
        self.mean_cb = np.array(mean[1], dtype=np.float32)
        self.std_cb = np.array(std[1], dtype=np.float32)
        self.mean_cr = np.array(mean[2], dtype=np.float32)
        self.std_cr = np.array(std[2], dtype=np.float32)
        self.to_rgb = to_rgb
        self.to_rgb_raw = to_rgb_raw
        self.inputnorm = inputnorm

    def __call__(self, results):
        results['dct_y'] = mmcv.imnormalize(results['dct_y'], self.mean_y, self.std_y, self.to_rgb)
        results['dct_cb'] = mmcv.imnormalize(results['dct_cb'], self.mean_cb, self.std_cb, self.to_rgb)
        results['dct_cr'] = mmcv.imnormalize(results['dct_cr'], self.mean_cr, self.std_cr, self.to_rgb)
        results['img_norm_cfg'] = dict(mean=[self.mean_y, self.mean_cb, self.mean_cr],
                                       std=[self.std_y, self.std_cb, self.std_cr], to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(mean={}, std={}, to_rgb={})'.format(
            self.mean_y, self.std_y, self.to_rgb)
        return repr_str

@PIPELINES.register_module
class NormalizeDCTUpscaledStatic(object):
    """Normalize the image.

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, channels=None, to_rgb=True, to_rgb_raw=False):

        self.to_rgb = to_rgb
        self.to_rgb_raw = to_rgb_raw
        self.channels = channels

        if channels == 192 or channels is None:
            self.mean = np.array(mean, dtype=np.float32)
            self.std = np.array(std, dtype=np.float32)
        else:
            subset_y  = dct_channel_index[channels][0]
            subset_cb = dct_channel_index[channels][1]
            subset_cb = [64+c for c in subset_cb]
            subset_cr = dct_channel_index[channels][2]
            subset_cr = [128+c for c in subset_cr]
            subset = subset_y + subset_cb + subset_cr
            self.mean, self.std = [mean[i] for i in subset], [std[i] for i in subset]
            self.mean = np.array(self.mean, dtype=np.float32)
            self.std = np.array(self.std, dtype=np.float32)

    def __call__(self, results):
        results['img'] = mmcv.imnormalize(results['img'], self.mean, self.std, self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(mean={}, std={}, to_rgb={})'.format(
            self.mean_y, self.std_y, self.to_rgb)
        return repr_str
