#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

from functools import partial

import torch
import torch.nn as nn

from pytorchvideo.losses.soft_target_cross_entropy import SoftTargetCrossEntropyLoss


class ContrastiveLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(ContrastiveLoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, dummy_labels=None):
        targets = torch.zeros(inputs.shape[0], dtype=torch.long).cuda()
        loss = nn.CrossEntropyLoss(reduction=self.reduction).cuda()(inputs, targets)
        return loss


class MultipleMSELoss(nn.Module):
    """
    Compute multiple mse losses and return their average.
    """

    def __init__(self, reduction="mean"):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(MultipleMSELoss, self).__init__()
        self.mse_func = nn.MSELoss(reduction=reduction)

    def forward(self, x, y):
        loss_sum = 0.0
        multi_loss = []
        for xt, yt in zip(x, y):
            if isinstance(yt, (tuple,)):
                if len(yt) == 2:
                    yt, wt = yt
                    lt = "mse"
                elif len(yt) == 3:
                    yt, wt, lt = yt
                else:
                    raise NotImplementedError
            else:
                wt, lt = 1.0, "mse"
            if lt == "mse":
                loss = self.mse_func(xt, yt)
            else:
                raise NotImplementedError
            loss_sum += loss * wt
            multi_loss.append(loss)
        return loss_sum, multi_loss


_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "soft_cross_entropy": partial(SoftTargetCrossEntropyLoss, normalize_targets=False),
    "contrastive_loss": ContrastiveLoss,
    "mse": nn.MSELoss,
    "multi_mse": MultipleMSELoss,
}


def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]


def build_loss_func_from_cfg(cfg, reduction="mean"):
    """根据配置构建损失函数。

    Args:
        cfg (CfgNode): 训练配置对象。
        reduction (str): 损失归约方式。

    Returns:
        nn.Module: 已实例化的损失函数。
    """
    loss_name = cfg.MODEL.LOSS_FUNC  # 中文注释：当前配置指定的损失名称。
    loss_builder = get_loss_func(loss_name)  # 中文注释：损失函数构造器。
    if loss_name != "cross_entropy" or not cfg.MODEL.CLASS_WEIGHTS:
        return loss_builder(reduction=reduction)

    class_weights = torch.tensor(  # 中文注释：按类别顺序排列的权重张量。
        cfg.MODEL.CLASS_WEIGHTS, dtype=torch.float32
    )
    if int(class_weights.numel()) != int(cfg.MODEL.NUM_CLASSES):
        raise ValueError(
            "MODEL.CLASS_WEIGHTS 长度必须与 MODEL.NUM_CLASSES 一致: "
            f"{class_weights.numel()} vs {cfg.MODEL.NUM_CLASSES}"
        )
    if cfg.NUM_GPUS:
        class_weights = class_weights.cuda()
    return loss_builder(weight=class_weights, reduction=reduction)
