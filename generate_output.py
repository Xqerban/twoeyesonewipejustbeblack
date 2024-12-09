import os
import json
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# 定义测试集的数据集类（与训练集相同，但不需要标签）
class TestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        """
        :param img_dir: 存放测试图像的文件夹路径
        :param transform: 需要应用于图像的变换（可选）
        """
        self.img_dir = img_dir
        self.transform = transform
        self.img_ids = os.listdir(img_dir)  # 获取所有图片的文件名
        self.original_sizes = {}

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, img_id)
        img = Image.open(img_path).convert("RGB")
        self.original_sizes[img_id] = img.size

        if self.transform:
            img = self.transform(img)

        return img, img_id  # 只返回图像和图片ID

# 加载模型
def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath))
    model.eval()  # 切换到评估模式
    print(f"模型已从 {filepath} 加载")
    return model

# 进行推理并生成结果
def generate_predictions(model, dataloader, output_json_path):
    results = []

    with torch.no_grad():  # 禁用梯度计算
        model.eval()  # 切换到评估模式
        for images, img_ids in tqdm(dataloader):
            images = [image.to(device) for image in images]

            # 模型推理
            predictions = model(images)


            for i, img_id in enumerate(img_ids):
                prediction = predictions[i]

                # 获取预测框和标签
                boxes = prediction['boxes'].cpu().numpy()
                labels = prediction['labels'].cpu().numpy()
                scores = prediction['scores'].cpu().numpy()

                # 只保留标签为1的框，即篡改的区域，阈值可根据需要调整
                mask = labels == 1
                boxes = boxes[mask]
                scores = scores[mask]
                region = []

                if len(boxes) > 0:
                    original_width, original_height = dataloader.dataset.original_sizes[img_id]  # 获取原始图像的大小
                    scale_x = original_width / 800  # 计算宽度的缩放比例
                    scale_y = original_height / 800  # 计算高度的缩放比例

                    # 将预测框的坐标缩放回原始图像的大小
                    boxes = boxes * np.array([scale_x, scale_y, scale_x, scale_y])  # 还原bbox大小

                    best_idx = np.argmax(scores)
                    best_box = boxes[best_idx]
                    best_score = scores[best_idx]

                    # 将最高置信度的框添加到结果中
                    region = [best_box.tolist()]  # 转换为原生类型

                # 如果有预测框，保存预测框
                # region = boxes.tolist() if len(boxes) > 0 else []

                # 将结果添加到列表中
                results.append({"id": img_id, "region": region})

    # 将结果保存为JSON文件
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"预测结果已保存到 {output_json_path}")

# 图像转换操作
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Resize((800, 800)),  # 统一图像大小
])

# 定义测试数据集和数据加载器
test_dataset = TestDataset(img_dir="data/image/val", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

model = fasterrcnn_resnet50_fpn(pretrained=False)

# 修改模型的分类头部分，将类别数改为2（篡改和未篡改）
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes=2)
# 修改模型的分类头部分，将类别数改为2（篡改和未篡改）
load_model(model, 'model/model.pth')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
generate_predictions(model, test_loader, "output/label_test.json")