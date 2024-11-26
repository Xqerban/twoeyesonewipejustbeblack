"""读取 COCO 格式的文件并将数据转换为 PyTorch 可用的格式"""
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
import torchvision.transforms as T
from pycocotools import mask
from torchvision import datasets

class CocoDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transforms=None):
        with open(annotation_file, 'r') as f:
            self.data = json.load(f)
        
        self.img_dir = img_dir
        self.transforms = transforms
        self.img_ids = [item['id'] for item in self.data['images']]

    def __len__(self):
        return len(self.data['images'])
    
    def __getitem__(self, idx):
        img_info = self.data['images'][idx]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")

        # 获取目标框（bounding boxes）和标签（labels）
        target = {}
        target['boxes'] = []
        target['labels'] = []

        for ann in self.data['annotations']:
            if ann['image_id'] == img_info['id']:
                # 如果图片中有目标（篡改的区域），则添加到目标框中
                target['boxes'].append(ann['bbox'])
                target['labels'].append(ann['category_id'])
        
        target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32)
        target['labels'] = torch.as_tensor(target['labels'], dtype=torch.int64)

        # 执行图像转换（可选）
        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

# 定义数据增强 TODO
def get_transform():
    return T.Compose([
        T.ToTensor(),
        T.Resize((800, 800)),  # TODO 根据需要调整大小
    ])
