import json
import os
from PIL import Image


def convert_to_yolo_format(json_file, img_dir, output_dir):
    # 读取标注文件
    with open(json_file, 'r') as f:
        annotations = json.load(f)

    # 遍历每个标注
    for anno in annotations:
        img_id = anno["id"]
        region = anno["region"]
        img_path = os.path.join(img_dir, img_id)

        # 打开图像，获取图像尺寸
        img = Image.open(img_path)
        img_width, img_height = img.size

        # 保存 YOLO 格式标注的路径
        yolo_txt_path = os.path.join(output_dir, img_id.replace(".jpg", ".txt"))

        # 如果没有标注区域，跳过
        if not region:
            continue

        # 打开文件写入 YOLO 格式标注
        with open(yolo_txt_path, 'w') as f:
            for box in region:
                # 计算 YOLO 格式的中心点坐标和宽高（归一化）
                xmin, ymin, xmax, ymax = box
                x_center = (xmin + xmax) / 2 / img_width
                y_center = (ymin + ymax) / 2 / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height

                # 写入 YOLO 格式的标注：类别、x_center、y_center、width、height
                f.write(f"1 {x_center} {y_center} {width} {height}\n")

    print(f"标注文件已保存到 {output_dir}")


# 输入参数
json_file = 'datasets/data/test.json'  # 你的 JSON 标注文件
img_dir = 'datasets/data/images/val'  # 图像文件夹
output_dir = 'datasets/data/labels/val'  # 保存 YOLO 格式标注文件的文件夹

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 执行转换
convert_to_yolo_format(json_file, img_dir, output_dir)
