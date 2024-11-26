import torch
from torch.utils.data import DataLoader
from dataset.dataset import CocoDataset
from dataset.dataset import build_transform
from config.config import Config
from dataset.coco import build_data

def main():
    train_ratio = Config.TRAIN_RATIO
    input_json = Config.TRAIN_LABEL
    train_dir = Config.IMG_DIR
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


if __name__ == "__main__":
    main()