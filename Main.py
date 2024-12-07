import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dataset.dataset import *
from tqdm import tqdm
from model.test import *
from train import *


# 加载数据集
train_dataset = CustomDataset(json_file=Config.TRAIN_ANN, img_dir=Config.TRAIN_IMG_DIR, transforms=transform)
train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
validation_dataset = CustomDataset(json_file=Config.VAL_ANN, img_dir=Config.TRAIN_IMG_DIR, transforms=transform)
validation_loader = DataLoader(validation_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
print("Dataset Loaded")

# 加载预训练模型并调整类别数量
print("Loaded model from {}".format(Config.MODEL_PATH))
model = fasterrcnn_resnet50_fpn(weights=None)
model.load_state_dict(torch.load(Config.MODEL_PATH))
print("Model Loaded")

# 获取输入通道数并调整类别数
in_features = model.roi_heads.box_predictor.cls_score.in_features
# 这里的 num_classes 是 2：背景 + 目标类
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=Config.NUM_CLASSES)

# 使用 CUDA 加速（如果可用）
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=Config.lr, momentum=0.9, weight_decay=Config.weight_decay)

train(model, device, train_loader, validation_loader, optimizer, Config.EPOCHS)
