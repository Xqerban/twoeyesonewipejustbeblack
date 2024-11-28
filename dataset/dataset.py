"""读取 COCO 格式的文件并将数据转换为 PyTorch 可用的格式"""
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pycocotools.coco import COCO
import os
from skimage import transform as sktsf
from PIL import Image
from utils.config import opt

class CocoDataset(Dataset):
    def __init__(self, json_file, img_dir, transform=None):
        self.coco = COCO(json_file)
        self.img_dir = img_dir
        self.transform = transform
        self.img_ids = list(self.coco.imgs.keys())
        
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")
        
        # 获取标注信息
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # 标注的 bbox 和类别
        boxes = []
        labels = []
        for ann in anns:
            boxes.append(ann['bbox'])
            labels.append(ann['category_id'])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
            'area': torch.tensor([ann['area'] for ann in anns]),
            'iscrowd': torch.zeros(len(anns), dtype=torch.int64)
        }
        
        if self.transform:
            img = self.transform(img)
        
        return img, target
    
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
