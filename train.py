import torch
from torch.utils.data import DataLoader
from dataset.dataset import CocoDataset
from dataset.dataset import build_transform
from config.config import Config

def main():
    transform = build_transform()
    
    train_dataset = CocoDataset(
        json_file=Config.TRAIN_ANN,
        img_dir=Config.IMG_DIR,
        transform=transform
    )
    val_dataset = CocoDataset(
        json_file=Config.VAL_ANN,
        img_dir=Config.IMG_DIR,
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))