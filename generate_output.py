from ultralytics import YOLO
import os
import json
from pathlib import Path
from PIL import Image
import tempfile

# 定义模型和数据路径
model_path = 'runs/train/yolo11_model/weights/best.pt'  # 你训练后保存的最佳模型路径
test_data_path = "data/image/val"  # 测试集图像文件夹路径
output_json_path = "output/predictions.json"  # 输出的预测结果文件路径

# 加载训练好的模型
model = YOLO(model_path)

# 获取测试集中的所有图片路径
image_paths = list(Path(test_data_path).glob("*.*"))  # 获取所有格式的图片（包括 PNG, JPEG, JPG）

# 保存预测结果
predictions = []


# 将图片转换为 JPG 格式
def convert_image_to_jpg(image_path):
    """
    读取图片并临时转换为 JPG 格式
    """
    with Image.open(image_path) as img:
        # 如果图片是 PNG 或其他格式，转换为 RGB 模式后返回 JPG 格式
        img = img.convert('RGB')

        # 使用临时文件保存转换后的 JPG 图片
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmpfile:
            tmpfile.close()  # 关闭文件，确保图片可以被 YOLO 读取
            img.save(tmpfile.name, format='JPEG')
            return tmpfile.name  # 返回临时文件的路径


# 对测试集进行预测
for image_path in image_paths:
    # 只处理 PNG 和 JPEG/JPG 格式的文件
    if image_path.suffix in ['.png', '.JPEG', '.jpg']:
        # 将图片临时转换为 JPG 格式
        tmp_jpg_path = convert_image_to_jpg(image_path)

        # 使用 YOLO 模型进行预测
        results = model.predict(tmp_jpg_path, imgsz=512)  # 传递文件路径

        # 提取预测结果
        pred = {
            "id": image_path.name,
            "region": []
        }

        # 结果通常会包含预测的边框坐标和置信度等信息
        for box in results[0].boxes:  # 获取所有框
            # 获取框的坐标（左上和右下角）
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # box.xyxy格式为 (x1, y1, x2, y2)
            pred["region"].append([x1, y1, x2, y2])

        # 将结果添加到列表中
        predictions.append(pred)

# 将预测结果保存为 JSON 格式
with open(output_json_path, 'w') as json_file:
    json.dump(predictions, json_file, indent=4)

print(f"预测结果已保存至 {output_json_path}")
