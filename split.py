import os
import json
import random
import shutil
from sklearn.model_selection import train_test_split

# 定义路径
img_dir = "data/image/train"  # 图片文件夹路径
json_file = "data/label_train.json"  # JSON 文件路径
train_img_dir = "datasets/data/images/train"  # 训练集图片文件夹
test_img_dir = "datasets/data/images/val"  # 测试集图片文件夹
train_json_file = "datasets/data/train.json"  # 训练集的 JSON 文件
test_json_file = "datasets/data/test.json"  # 测试集的 JSON 文件

# 创建训练集和测试集文件夹
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(test_img_dir, exist_ok=True)

# 读取原始的 JSON 文件
with open(json_file, 'r') as f:
    annotations = json.load(f)

# 提取图片文件名列表
img_ids = [anno["id"] for anno in annotations]

# 使用 sklearn 的 train_test_split 按 9:1 的比例划分数据集
train_ids, test_ids = train_test_split(img_ids, test_size=0.1, random_state=42)

# 按照划分的 ID 生成训练集和测试集的图片列表
train_annotations = [anno for anno in annotations if anno["id"] in train_ids]
test_annotations = [anno for anno in annotations if anno["id"] in test_ids]

# 将训练集和测试集的图片复制到新的文件夹中
for anno in train_annotations:
    img_path = os.path.join(img_dir, anno["id"])
    new_img_path = os.path.join(train_img_dir, anno["id"])
    shutil.copy(img_path, new_img_path)

for anno in test_annotations:
    img_path = os.path.join(img_dir, anno["id"])
    new_img_path = os.path.join(test_img_dir, anno["id"])
    shutil.copy(img_path, new_img_path)

# 保存训练集和测试集的 JSON 文件
with open(train_json_file, 'w') as f:
    json.dump(train_annotations, f, indent=4)

with open(test_json_file, 'w') as f:
    json.dump(test_annotations, f, indent=4)

print(f"数据集已按 9:1 划分，训练集和测试集的图片与标注已保存。")
