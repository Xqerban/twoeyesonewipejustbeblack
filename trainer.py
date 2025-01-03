from __future__ import absolute_import

import os
import time
from collections import namedtuple

import torch
import torch as t
from torch import nn
from torch.nn import functional as F
from torchnet.meter import AverageValueMeter, ConfusionMeter

from model.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator
from utils import array_tool as at
from utils.config import opt
from utils.vis_tool import Visualizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LossTuple = namedtuple(
    "LossTuple",
    ["rpn_loc_loss", "rpn_cls_loss", "roi_loc_loss", "roi_cls_loss", "total_loss"],
)


# 封装Faster R-CNN的训练过程
class FasterRCNNTrainer(nn.Module):
    """wrapper for conveniently training. return losses

    The losses include:

    * :obj:`rpn_loc_loss`: The localization loss for \
        Region Proposal Network (RPN).
    * :obj:`rpn_cls_loss`: The classification loss for RPN.
    * :obj:`roi_loc_loss`: The localization loss for the head module.
    * :obj:`roi_cls_loss`: The classification loss for the head module.
    * :obj:`total_loss`: The sum of 4 loss above.

    Args:
        faster_rcnn (model.FasterRCNN):
            A Faster R-CNN model that is going to be trained.
    """

    def __init__(self, faster_rcnn):
        super(FasterRCNNTrainer, self).__init__()

        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = opt.rpn_sigma
        self.roi_sigma = opt.roi_sigma

        # target creator create gt_bbox gt_label etc as training targets.
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std

        self.optimizer = self.faster_rcnn.get_optimizer()
        # visdom wrapper
        self.vis = Visualizer(env=opt.env)

        # indicators for training status
        self.rpn_cm = ConfusionMeter(2)
        self.roi_cm = ConfusionMeter(21)
        self.meters = {
            k: AverageValueMeter() for k in LossTuple._fields
        }  # average loss

    def forward(self, imgs, bboxes, labels, scale):
        """前向传播

        Here are notations used.

        * :math:`N` is the batch size.
        * :math:`R` is the number of bounding boxes per image.

        Currently, only :math:`N=1` is supported.

        Args:
            imgs (~torch.autograd.Variable): A variable with a batch of images.
            bboxes (~torch.autograd.Variable): A batch of bounding boxes.
                Its shape is :math:`(N, R, 4)`.
            labels (~torch.autograd..Variable): A batch of labels.
                Its shape is :math:`(N, R)`. The background is excluded from
                the definition, which means that the range of the value
                is :math:`[0, L - 1]`. :math:`L` is the number of foreground
                classes.
            scale (float): Amount of scaling applied to
                the raw image during preprocessing.

        Returns:
            namedtuple of 5 losses
        """
        n = bboxes.shape[0]
        if n != 1:
            raise ValueError("Currently only batch size 1 is supported.")

        _, _, H, W = imgs.shape
        img_size = (H, W)

        # Extract features from the input images.
        features = self.faster_rcnn.extractor(imgs)
        # Use RPN to generate proposals.
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.faster_rcnn.rpn(
            features, img_size, scale
        )

        # Since batch size is one, convert variables to singular form
        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois

        # Sample RoIs and forward
        # It's fine to break the computation graph of rois,
        # consider them as constant input

        # Generate proposal targets.
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            at.tonumpy(bbox),
            at.tonumpy(label),
            self.loc_normalize_mean,
            self.loc_normalize_std,
        )
        # It's all zero because now it only support for batch=1 now
        sample_roi_index = t.zeros(len(sample_roi))

        # Head forward
        roi_cls_loc, roi_score = self.faster_rcnn.head(
            features, sample_roi, sample_roi_index
        )

        # ------------------ RPN losses -------------------#
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            at.tonumpy(bbox), anchor, img_size
        )
        gt_rpn_label = at.totensor(gt_rpn_label).long()
        gt_rpn_loc = at.totensor(gt_rpn_loc)

        # NOTE  Use `_fast_rcnn_loc_loss` to calculate the localization loss.
        # rpn_loc_loss = ...
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_loc, gt_rpn_loc, gt_rpn_label.data, self.rpn_sigma
        )

        # NOTE  Use `F.cross_entropy` to calculate the classification loss.
        # rpn_cls_loss = ...
        rpn_cls_loss = F.cross_entropy(
            rpn_score, gt_rpn_label.to(device), ignore_index=-1
        )

        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = at.tonumpy(rpn_score)[at.tonumpy(gt_rpn_label) > -1]
        self.rpn_cm.add(at.totensor(_rpn_score, False), _gt_rpn_label.data.long())

        # ------------------ ROI losses (fast rcnn loss) -------------------#
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[
            t.arange(0, n_sample).long().to(device), at.totensor(gt_roi_label).long()
        ]
        gt_roi_label = at.totensor(gt_roi_label).long()
        gt_roi_loc = at.totensor(gt_roi_loc)

        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc.contiguous(), gt_roi_loc, gt_roi_label.data, self.roi_sigma
        )
        roi_cls_loss = F.cross_entropy(roi_score, gt_roi_label.to(device))

        self.roi_cm.add(at.totensor(roi_score, False), gt_roi_label.data.long())

        # NOTE  Calculate all losses.
        # losses += ...
        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)]

        return LossTuple(*losses)

    def train_step(self, imgs, bboxes, labels, scale):
        """
        执行一次训练步骤，包括前向传播、反向传播和参数更新
        """
        # NOTE Main training step.
        # Zero gradients, forward, backward, and update.
        # Att: `losses` is a namedtuple of 5 losses. We use `losses.total_loss` to do backward.
        # self.optimizer ... (zero_grad)
        # losses = self.forward(...)
        # losses.total_loss.backward()
        # self.optimizer ... (step)
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, labels, scale)
        losses.total_loss.backward()
        self.optimizer.step()

        # Update the dict of meters, log the loss
        self.update_meters(losses)
        return losses

    def save(self, save_optimizer=False, save_path=None, **kwargs):
        """serialize models include optimizer and other info 保存模型状态
        return path where the model-file is stored.

        Args:
            save_optimizer (bool): whether save optimizer.state_dict().
            save_path (string): where to save model, if it's None, save_path
                is generate using time str and info from kwargs.

        Returns:
            save_path(str): the path to save models.
        """
        save_dict = dict()

        save_dict["model"] = self.faster_rcnn.state_dict()
        save_dict["config"] = opt._state_dict()
        save_dict["other_info"] = kwargs
        save_dict["vis_info"] = self.vis.state_dict()

        if save_optimizer:
            save_dict["optimizer"] = self.optimizer.state_dict()

        if save_path is None:
            timestr = time.strftime("%m%d%H%M")
            save_path = "checkpoints/fasterrcnn_%s" % timestr
            for k_, v_ in kwargs.items():
                save_path += "_%s" % v_

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        t.save(save_dict, save_path)
        self.vis.save([self.vis.env])
        return save_path

    def load(
        self,
        path,
        load_optimizer=True,
        parse_opt=False,
    ):
        state_dict = t.load(path)
        if "model" in state_dict:
            self.faster_rcnn.load_state_dict(state_dict["model"])
        else:  # legacy way, for backward compatibility
            self.faster_rcnn.load_state_dict(state_dict)
            return self
        if parse_opt:
            opt._parse(state_dict["config"])
        if "optimizer" in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict["optimizer"])
        return self

    def update_meters(self, losses):
        loss_d = {k: at.scalar(v) for k, v in losses._asdict().items()}
        for key, meter in self.meters.items():
            meter.add(loss_d[key])

    def reset_meters(self):
        """重置所有评估指标
        通常在每个epoch开始时调用,以便重新开始统计新的epoch的指标。
        """
        for key, meter in self.meters.items():
            meter.reset()
        self.roi_cm.reset()
        self.rpn_cm.reset()

    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}


# 损失函数
def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma**2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1.0 / sigma2)).float()
    y = flag * (sigma2 / 2.0) * (diff**2) + (1 - flag) * (abs_diff - 0.5 / sigma2)
    return y.sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = t.zeros(gt_loc.shape).to(device)

    # Localization loss is calculated only for positive rois.
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).to(device)] = 1
    # Apply smooth L1 loss
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= (gt_label >= 0).sum().float()  # ignore gt_label==-1 for rpn_loss

    return loc_loss
