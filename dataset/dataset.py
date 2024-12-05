import torch as t
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from utils.config import opt
from dataset.coco_dataset import CocoDataset
from . import util
import numpy as np
import torch


class Dataset(object):
    def __init__(self, label_path, img_dir):
        self.db = CocoDataset(label_path, img_dir)
        self.tsf = Transform(opt.min_size, opt.max_size)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        if ori_img is None or len(bbox) == 0 or len(label) == 0:
            return None
        img, bbox, label, scale = self.tsf((ori_img, bbox, label))
        return img.copy(), bbox.clone(), label.clone(), scale

    def __len__(self):
        return len(self.db)
    
class TestDataset(object):
    def __init__(self, label_path, img_dir, split="test", use_difficult=True):# TODO false or true
        self.db = CocoDataset(
            label_path, img_dir, use_difficult=use_difficult
        )

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        img = preprocess(ori_img)
        return img, ori_img.shape[1:], bbox, label, difficult

    def __len__(self):
        return len(self.db)


class Transform(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bbox, label = in_data
        H, W, _ = img.shape
        img = preprocess(img, self.min_size, self.max_size)
        o_H, o_W, _ = img.shape
        scale = o_H / H
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))

        # horizontally flip
        img, params = util.random_flip(img, x_random=True, return_param=True)
        bbox = util.flip_bbox(bbox, (o_H, o_W), x_flip=params["x_flip"])

        return img, bbox, label, scale


def inverse_normalize(img):
    """逆归一化
    """
    if opt.caffe_pretrain:
        img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
        return img[::-1, :, :]
    # approximate un-normalize for visualize
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255


def pytorch_normalze(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = normalize(t.from_numpy(img))
    return img.numpy()


def caffe_normalize(img):
    """
    return appr -125-125 BGR
    """
    img = img[[2, 1, 0], :, :]  # RGB-BGR
    img = img * 255
    mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
    img = (img - mean).astype(np.float32, copy=True)
    return img

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
    H, W, C = img.shape
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


def collate_fn(batch):
    # 过滤掉 None 样本
    batch = [item for item in batch if item[0] is not None]

    # 使用默认的 collate_fn 进行批处理
    images, bboxes, labels, scales = zip(*batch)
    images = torch.stack(images, dim=0)
    bboxes = torch.stack(bboxes, dim=0)
    labels = torch.stack(labels, dim=0)

    return images, bboxes, labels, scales
