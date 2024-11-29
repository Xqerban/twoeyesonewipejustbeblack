import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pycocotools.coco import COCO

from PIL import Image
import os
import numpy as np


class CocoDataset(Dataset):
    def __init__(self, json_file, data_dir, use_difficult=False, return_difficult=False):
        self.coco = COCO(json_file)
        self.data_dir = data_dir
        self.img_ids = list(self.coco.imgs.keys())
        self.label_names = COCO_BBOX_LABEL_NAMES
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult

    def __len__(self):
        return len(self.img_ids)

    def get_example(self, idx):
        """获取数据集中指定索引的样本

        Args:
            idx (int): 样本索引

        Returns:
            tuple: 包含以下元素的元组:
                - img (ndarray): 图像数组
                - boxes (Tensor): 边界框坐标 [ymin, xmin, ymax, xmax]
                - labels (Tensor): 类别标签
                - difficult (Tensor): 是否为困难样本的标志
                
                如果样本无效(无boxes)则返回 None
        """
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.data_dir, img_info['file_name'])
        img = Image.open(img_path)
        img = np.array(img)

        # 获取标注信息
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # 标注的 bbox 和类别
        boxes = []
        labels = []
        difficult = []

        for ann in anns:
            if not self.use_difficult and ann.get('difficult', 0) == 1:
                continue

            # 如果 bbox 为空，跳过该标注
            if not ann['bbox']:
                continue

            # Convert COCO bbox [xmin, ymin, width, height] to [ymin, xmin, ymax, xmax]
            xmin, ymin, width, height = ann['bbox']
            boxes.append([ymin, xmin, ymin + height, xmin + width])
            labels.append(ann['category_id']) 
            difficult.append(ann.get('difficult', 0))


        # 如果 boxes 为空，返回 None    
        if not boxes:
            return None

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int32)
        difficult = torch.tensor(difficult, dtype=torch.uint8)  # Use uint8 for difficult flags

        return img, boxes, labels, difficult

    __getitem__ = get_example


COCO_BBOX_LABEL_NAMES = (
    "tampered",
    "untampered",
)



