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

COCO_BBOX_LABEL_NAMES = (
    "tampered",
    "untampered",
)