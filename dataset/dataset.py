"""读取 COCO 格式的文件并将数据转换为 PyTorch 可用的格式"""
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pycocotools.coco import COCO
import os
from skimage import transform as sktsf
from PIL import Image
from utils.config import opt

    
def build_transform():
    # TODO
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform


def preprocess(img, min_size=600, max_size=1000):
    """Preprocess an image for feature extraction.

    The length of the shorter edge is scaled to :obj:`self.min_size`.
    After the scaling, if the length of the longer edge is longer than
    :param min_size:
    :obj:`self.max_size`, the image is scaled to fit the longer edge
    to :obj:`self.max_size`.

    After resizing the image, the image is subtracted by a mean image value
    :obj:`self.mean`.

    Args:
        img (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.

    Returns:
        ~numpy.ndarray: A preprocessed image.

    """
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.0
    img = sktsf.resize(
        img, (C, H * scale, W * scale), mode="reflect", anti_aliasing=False
    )
    # both the longer and shorter should be less than
    # max_size and min_size
    if opt.caffe_pretrain:
        normalize = caffe_normalize
    else:
        normalize = pytorch_normalze
    return normalize(img)
