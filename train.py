import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.dataset import CocoDataset
from dataset.dataset import build_transform
from utils.config import Config
from dataset.coco import build_data
from model.faster_rcnn_vgg16 import *

def train():
    train_ratio = Config.TRAIN_RATIO
    input_json = Config.TRAIN_LABEL
    train_dir = Config.TRAIN_IMG_DIR
    train_labels_path = Config.TRAIN_ANN
    val_labels_path = Config.VAL_ANN

    build_data(train_ratio, input_json, train_dir, train_labels_path, val_labels_path)

    transform = build_transform()
    
    train_dataset = CocoDataset(
        json_file=train_labels_path,
        img_dir=train_dir,
        transform=transform
    )
    val_dataset = CocoDataset(
        json_file=val_labels_path,
        img_dir=train_dir,
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # TODO
    model = FasterRCNNVGG16(n_fg_class=Config.NUM_CLASSES)
    model.to(Config.DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=Config.LEARNING_RATE)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(Config.EPOCHS):
        model.train()
        for images, targets in train_loader:
            images = [image.to(Config.DEVICE) for image in images]
            targets = [{k: v.to(Config.DEVICE) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        lr_scheduler.step()
        
        print(f"Epoch [{epoch+1}/{Config.EPOCHS}], Loss: {losses.item()}")

    torch.save(model.state_dict(), './checkpoints/model.pth')


if __name__ == "__main__":
    train()
