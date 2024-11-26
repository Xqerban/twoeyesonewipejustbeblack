"""读取 COCO 格式的文件并将数据转换为 PyTorch 可用的格式"""
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
import torchvision.transforms as T

class CocoDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transforms=None):
        with open(annotation_file, 'r') as f:# COCO 格式的标注文件（JSON 格式）
            self.data = json.load(f)
        
        self.img_dir = img_dir # 图片目录
        self.transforms = transforms
        self.img_ids = [item['id'] for item in self.data['images']]

    def __len__(self):
        return len(self.data['images'])
    
    def __getitem__(self, idx):
        img_info = self.data['images'][idx]
        img_path = os.path.join(self.img_dir, img_info['file_name']) # 图片的完整路径
        img = Image.open(img_path).convert("RGB")

        target = {}
        target['boxes'] = []
        target['labels'] = []
        target['area'] = []

        for annotation in self.data['annotations']:
            if annotation['image_id'] == img_info['id']:
                target['boxes'].append(annotation['bbox'])
                target['labels'].append(annotation['category_id'])
                target['area'].append(annotation['area'])
        
        target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32)
        target['labels'] = torch.as_tensor(target['labels'], dtype=torch.int64)
        target['area'] = torch.as_tensor(target['area'], dtype=torch.float32)

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target
    
# TODO
def get_transform():
    return T.Compose([
        T.ToTensor(),
        T.Resize((800, 800)),
    ])
