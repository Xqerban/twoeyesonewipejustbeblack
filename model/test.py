import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from dataset.dataset import CustomDataset  # 假设你有自己的自定义数据集
from tqdm import tqdm
import numpy as np

# 计算 mAP
def compute_metrics(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_threshold=0.5):
    # 计算 Precision, Recall, F1-score
    def compute_iou(box1, box2):
        x1, y1, x2, y2 = box1
        x1_ = box2[0]
        y1_ = box2[1]
        x2_ = box2[2]
        y2_ = box2[3]

        # 计算交集区域
        xi1 = max(x1, x1_)
        yi1 = max(y1, y1_)
        xi2 = min(x2, x2_)
        yi2 = min(y2, y2_)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        # 计算并集区域
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_ - x1_) * (y2_ - y1_)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    # 计算每个图像的 Precision, Recall, F1-score
    true_positives = []
    false_positives = []
    false_negatives = []

    for i in range(len(gt_boxes)):
        gt_box = gt_boxes[i]
        gt_label = gt_labels[i]

        best_iou = 0
        best_match = -1

        for j in range(len(pred_boxes)):
            if pred_labels[j] == gt_label:
                iou = compute_iou(gt_box, pred_boxes[j])
                if iou > best_iou:
                    best_iou = iou
                    best_match = j

        if best_iou >= iou_threshold and best_match != -1:
            true_positives.append(pred_scores[best_match])
            false_positives.append(0)
        else:
            false_negatives.append(1)

    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(false_negatives)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1


def compute_mAP(pred_boxes_all, pred_labels_all, pred_scores_all, gt_boxes_all, gt_labels_all, iou_threshold=0.5):
    precisions, recalls, f1s = [], [], []

    for pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels in zip(pred_boxes_all, pred_labels_all,
                                                                         pred_scores_all, gt_boxes_all, gt_labels_all):
        precision, recall, f1 = compute_metrics(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels,
                                                iou_threshold)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    mAP = np.mean(precisions)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1s)

    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall: {avg_recall:.4f}")
    print(f"F1-score: {avg_f1:.4f}")
    print(f"mAP: {mAP:.4f}")

    return avg_precision, avg_recall, avg_f1, mAP

def evaluate(model, data_loader, device):
    model.eval()

    # 使用 CUDA 加速（如果可用）
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # 存储所有预测和真实数据
    pred_boxes_all = []
    pred_labels_all = []
    pred_scores_all = []
    gt_boxes_all = []
    gt_labels_all = []

    # 推理阶段
    with torch.no_grad():
        for images, targets in tqdm(data_loader):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 获取模型预测结果
            prediction = model(images)

            # 遍历每张图片
            for i in range(len(prediction)):
                pred_boxes = prediction[i]['boxes'].cpu().numpy()  # Predicted bounding boxes
                pred_labels = prediction[i]['labels'].cpu().numpy()  # Predicted labels
                pred_scores = prediction[i]['scores'].cpu().numpy()  # Predicted scores

                gt_boxes = targets[i]['boxes'].cpu().numpy()  # Ground truth boxes
                gt_labels = targets[i]['labels'].cpu().numpy()  # Ground truth labels

                # 将预测和真实数据添加到列表
                pred_boxes_all.append(pred_boxes)
                pred_labels_all.append(pred_labels)
                pred_scores_all.append(pred_scores)
                gt_boxes_all.append(gt_boxes)
                gt_labels_all.append(gt_labels)

    # 计算mAP
    compute_mAP(pred_boxes_all, pred_labels_all, pred_scores_all, gt_boxes_all, gt_labels_all)

