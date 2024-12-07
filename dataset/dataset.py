from torch.utils.data import DataLoader, Dataset
import json
import os
from PIL import Image
import torch
from torchvision import transforms
from config import Config


class CustomDataset(Dataset):
    def __init__(self, json_file, img_dir, transforms=None):
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        self.img_dir = img_dir
        self.transforms = transforms

        # 假设目标类的 ID 为 1，背景是 0
        self.target_class_id = 1

    def __len__(self):
        return len(self.data['images'])

    def __getitem__(self, idx):
        img_info = self.data['images'][idx]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")

        # 获取当前图片的所有标注
        annotations = [ann for ann in self.data['annotations'] if ann['image_id'] == img_info['id']]

        # 准备目标框
        boxes = []
        labels = []
        for ann in annotations:
            if ann['category_id'] == self.target_class_id:  # 仅保留目标类
                bbox = ann['bbox']  # 例如 [x, y, width, height]
                if bbox[2] > img.size[0] or bbox[3] > img.size[1] or bbox[0] < 0 or bbox[1] < 0 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    continue
                bbox = resize_bbox(bbox, img.size)
                boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])  # 转换为 [xmin, ymin, xmax, ymax]
                labels.append(ann['category_id'])

        # 转为 tensor
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # 创建目标字典
        target = {'boxes': boxes, 'labels': labels}

        if self.transforms:
            img = self.transforms(img)

        return img, target


def collate_fn(batch):
    images, targets = zip(*batch)

    # 处理空目标框的情况
    for i in range(len(targets)):
        if targets[i]['boxes'].shape[0] == 0:  # 如果没有目标框
            targets[i]['boxes'] = torch.zeros((0, 4))  # 使用空框
            targets[i]['labels'] = torch.tensor([0], dtype=torch.int64)
        else:
            targets[i]['labels'] = targets[i]['labels'].to(torch.int64)

            # 处理图像堆叠
    images = torch.stack([image for image in images], 0)

    return images, targets


def resize_bbox(bbox, img_size):
    W, H = img_size

    # 设置目标尺寸
    new_size = Config.IMG_SIZE

    # 计算缩放比例
    scale_x = new_size / W
    scale_y = new_size / H

    bbox = torch.tensor([
        bbox[0] * scale_x,  # xmin 缩放
        bbox[1] * scale_y,  # ymin 缩放
        bbox[2] * scale_x,  # xmax 缩放
        bbox[3] * scale_y  # ymax 缩放
    ])
    return bbox


transform = transforms.Compose([
    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
    transforms.ToTensor()
])
