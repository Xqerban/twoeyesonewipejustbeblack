from __future__ import absolute_import

import torch as t
from torch import nn
from torchvision.models import vgg16
from torchvision.ops import RoIPool

from model.faster_rcnn import FasterRCNN
from model.region_proposal_network import RegionProposalNetwork
from utils import array_tool as at


def decom_vgg16():
    # `model` is a Dict containing `features` and `classifier`.
    model = vgg16(True)

    # NOTE  Only use the first 30 layers of the model,
    # `features` is a list of layers.
    features = list(model.features)[:30]
    features = nn.Sequential(*features)

    # Classifier will be used in `VGG16RoIHead`
    classifier = model.classifier
    classifier = list(classifier)
    # delete dropout
    del classifier[6]
    del classifier[5]
    del classifier[2]
    # `classifier` is a list of layers, convert it to a Sequential.
    classifier = nn.Sequential(*classifier)

    # freeze top4 conv
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False

    # NOTE  Return a Sequential of layers and a Sequential of classifier layers.
    return features, classifier


class FasterRCNNVGG16(FasterRCNN):
    """Faster R-CNN based on VGG-16.
    For descriptions on the interface of this model, please refer to
    :class:`model.faster_rcnn.FasterRCNN`.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.
    """

    feat_stride = 16  # downsample 16x for output of conv5 in vgg16

    def __init__(self, n_fg_class, ratios, anchor_scales):
        """初始化 Faster R-CNN VGG16 模型

        Args:
            n_fg_class (int): 不包括背景的类别数量
            ratios (list of floats): anchor 的宽高比列表
            anchor_scales (list of numbers): anchor 的尺度列表,
                最终的 anchor 面积将是 anchor_scales 中元素的平方与原始窗口面积的乘积
        """
        # NOTE  Decompose VGG16 model
        extractor, classifier = decom_vgg16()

        # NOTE  Init RPN and Head
        rpn = RegionProposalNetwork(
            in_channels=512,  
            mid_channels=512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        )

        # NOTE  Init RoIHead
        head = VGG16RoIHead(
            n_class=n_fg_class + 1,  
            roi_size=7,
            spatial_scale=(1.0 / self.feat_stride),
            classifier=classifier,
        )

        super(FasterRCNNVGG16, self).__init__(
            extractor,
            rpn,
            head,
        )


class VGG16RoIHead(nn.Module):
    """Faster R-CNN Head for VGG-16 based implementation.
    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.

    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        classifier (nn.Module): Two layer Linear ported from vgg16

    """

    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        # n_class includes the background
        super(VGG16RoIHead, self).__init__()

        self.classifier = classifier  # 7 * 7 * 512 -> 4096

        # NOTE  Init linear layer to get the class location.
        # 4096 -> n_class * 4
        self.cls_loc = nn.Linear(4096, n_class * 4)  

        # NOTE  Init linear layer to get the class score.
        # 4096 -> n_class
        self.score = nn.Linear(4096, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale

        # NOTE  Init ROIPool
        # Att: `roi_size, spatial_scale` are parameters of `RoIPool.__init__`
        self.roi = RoIPool(roi_size, spatial_scale)

    def forward(self, x, rois, roi_indices):
        """Forward the chain.

        We assume that there are :math:`N` batches.

        Args:
            x (Variable): 4D image variable.
            rois (Tensor): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (Tensor): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.
        """
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)

        # Important: (y1, x1, y2, x2) -> (x1, y1, x2, y2)
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        # Make sure the tensor is contiguous, good for performance.
        indices_and_rois = xy_indices_and_rois.contiguous()

        # NOTE  Apply RoIPool
        pool = self.roi(x, indices_and_rois)

        # Flatten the output of the RoIPool layer,
        # so that it can be fed into the classifier.
        pool = pool.view(pool.size(0), -1)

        # NOTE  Apply the classifier and predict the class and location.
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)

        return roi_cls_locs, roi_scores


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(
            mean
        )  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
