from config import Config
import torch
from tqdm import tqdm
from model.test import evaluate


def train(model, device, train_loader, validation_loader, optimizer, epoch):
    for epoch in range(epoch):
        model.train()
        running_loss = 0.0
        for images, targets in tqdm(train_loader):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 前向和反向传播
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # 清空梯度
            optimizer.zero_grad()

            # 反向传播
            losses.backward()

            # 更新权重
            optimizer.step()

            running_loss += losses.item()

        print(f"Epoch [{epoch + 1}/{epoch}], Loss: {running_loss / len(train_loader)}")
        torch.save(model.state_dict(), "faster_rcnn_target_model.pth")
        if (epoch + 1) % Config.VAL_EVERY_N_EPOCHS == 0:  # 每训练几轮进行一次验证
            print("Evaluating model...")
            mAP = evaluate(model, validation_loader, device)