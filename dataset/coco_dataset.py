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

            # Convert COCO bbox [xmin, ymin, width, height] to [ymin, xmin, ymax, xmax]
            xmin, ymin, width, height = ann['bbox']
            boxes.append([ymin, xmin, ymin + height, xmin + width])
            labels.append(ann['category_id'])
            difficult.append(ann.get('difficult', 0))

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int32)
        difficult = torch.tensor(difficult, dtype=torch.uint8)  # Use uint8 for difficult flags

        # if self.return_difficult:
        #     return img, boxes, labels, difficult
        return img, boxes, labels, difficult

    __getitem__ = get_example


COCO_BBOX_LABEL_NAMES = (
    "tampered",
    "untampered",
)



