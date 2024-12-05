import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.dataset import CocoDataset
from dataset.dataset import Transform
from utils.config import Config
from dataset.coco import build_data
from model.faster_rcnn_vgg16 import *
from utils.config import opt
from dataset.dataset import *
import os
from trainer import FasterRCNNTrainer

def train(**kwargs):
    opt._parse(kwargs)

    train_ratio = opt.TRAIN_RATIO
    input_json = opt.TRAIN_LABEL
    train_dir = opt.TRAIN_IMG_DIR
    train_labels_path = opt.TRAIN_ANN
    val_labels_path = opt.VAL_ANN

    # 如果训练数据和验证数据不存在，则生成数据
    if not os.path.exists(train_labels_path) or not os.path.exists(val_labels_path):
        build_data(train_ratio, input_json, train_dir, train_labels_path, val_labels_path)

    print("load data...")
    train_dataset = Dataset(train_labels_path, train_dir)
    val_dataset = TestDataset(val_labels_path, train_dir)

    train_loader = DataLoader(train_dataset, batch_size=opt.BATCH_SIZE, shuffle=True, collate_fn=collate_fn,num_workers=opt.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=opt.BATCH_SIZE, shuffle=False, collate_fn=collate_fn,num_workers=opt.test_num_workers)

    # TODO
    model = FasterRCNNVGG16(n_fg_class=opt.NUM_CLASSES, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32])
    print("model construct completed")

    trainer = FasterRCNNTrainer(model).to(opt.DEVICE)
                
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=opt.LEARNING_RATE)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(opt.EPOCHS):
        model.train()
        for images, targets in train_loader:
            images = [image.to(opt.DEVICE) for image in images]
            targets = [{k: v.to(opt.DEVICE) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        lr_scheduler.step()
        
        print(f"Epoch [{epoch+1}/{opt.EPOCHS}], Loss: {losses.item()}")

    torch.save(model.state_dict(), './checkpoints/model.pth')


if __name__ == "__main__":
    train()
