import torch
import torch.nn as nn

from mmdet.core import bbox2result
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector


@DETECTORS.register_module
class SingleStageDetector(BaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super(SingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat_dct(self, dct_y, dct_cb, dct_cr, dynamic_input=False):
        if isinstance(dynamic_input, list): dynamic_input = dynamic_input[0]
        if not dynamic_input:
            x = self.backbone(dct_y, dct_cb, dct_cr)
        else:
            x, inp_atten = self.backbone(dct_y, dct_cb, dct_cr)
        if self.with_neck:
            x = self.neck(x)

        if not dynamic_input:
            return x
        else:
            return x, inp_atten

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train_dct(self, dct_y, dct_cb, dct_cr, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None,
                          dynamic_input=True):
        if not dynamic_input:
            x = self.extract_feat_dct(dct_y, dct_cb, dct_cr, dynamic_input)
        else:
            x, gate_activations = self.extract_feat_dct(dct_y, dct_cb, dct_cr, dynamic_input)

        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        if dynamic_input:
            acts = torch.tensor([0.]).cuda()
            for ga in gate_activations:
                acts += torch.mean(ga)
            loss_gate_activations = acts / len(gate_activations)
            losses.update(dict(loss_gate_activations=loss_gate_activations))

        return losses

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test_dct(self, dct_y, dct_cb, dct_cr, img_meta, rescale=False):
        dynamic_input = True if 'dynamic_input' in img_meta[0] else False

        if not dynamic_input:
            x = self.extract_feat_dct(dct_y, dct_cb, dct_cr, dynamic_input=dynamic_input)
        else:
            x, gate_activations = self.extract_feat_dct(dct_y, dct_cb, dct_cr, dynamic_input=dynamic_input)

        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]

        if not dynamic_input:
            return bbox_results[0]
        else:
            return bbox_results[0], gate_activations

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError

    def aug_test_dct(self, dct_y, dct_cb, dct_cr, img_metas, rescale=False):
        raise NotImplementedError