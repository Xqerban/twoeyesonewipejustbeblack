import json
import os
from PIL import Image
from sklearn.model_selection import train_test_split

def convert_to_coco(annotations, train_dir):
    coco_dataset = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "tampered"},
            {"id": 1, "name": "untampered"}
        ]
    }

    annotation_id = 1
    for item in annotations:
        img_id = item['id']
        img_path = os.path.join(train_dir, img_id)
        # 提取文件名中的数字部分作为图片的唯一 ID
        img_numeric_id = int(img_id.split('_')[1].split('.')[0])

        # 获取图片的宽度和高度
        if os.path.exists(img_path):
            with Image.open(img_path) as img:
                width, height = img.size
        
        # images
        img_info = {
            "id": img_numeric_id,
            "file_name": img_id,
            "height": height,  
            "width": width
        }
        coco_dataset["images"].append(img_info)
        
        # annotations
        if item['region']:
            # 有篡改的情况，类别为 tampered
            for region in item['region']:
                x1, y1, x2, y2 = region
                annotation = {
                    "id": annotation_id,
                    "image_id": img_numeric_id,
                    "category_id": 1,  # tampered 类别
                    "bbox": [x1, y1, x2-x1, y2-y1], 
                    "area": (x2-x1)*(y2-y1),
                    "iscrowd": 0
                }
                coco_dataset["annotations"].append(annotation)
                annotation_id += 1
        else:
            # 没有篡改的情况，类别为 untampered
            annotation = {
                "id": annotation_id,
                "image_id": img_numeric_id,
                "category_id": 2,  # untampered 类别
                "bbox": [],  # 无篡改，不需要边界框
                "area": 0,  # 无边界框，面积为 0
                "iscrowd": 0
            }
            coco_dataset["annotations"].append(annotation)
            annotation_id += 1
    
    return coco_dataset

# 划分数据集
def split_dataset(annotations, train_ratio):
    all_images = [item['id'] for item in annotations]
    # 确保数据被打乱
    train_images, val_images = train_test_split(all_images, train_size=train_ratio, shuffle=True, random_state=42)

    train_annotations = [item for item in annotations if item['id'] in train_images]
    val_annotations = [item for item in annotations if item['id'] in val_images]
    
    return train_annotations, val_annotations

def build_data(train_ratio, input_json, train_dir, train_labels_path, val_labels_path):
    with open(input_json, 'r') as f:
        original_annotations = json.load(f)

    # 划分训练集和验证集
    train_annotations, val_annotations = split_dataset(original_annotations, train_ratio)

    # 转换并保存COCO格式数据集
    train_coco = convert_to_coco(train_annotations, train_dir)
    val_coco = convert_to_coco(val_annotations, train_dir)

    # 保存训练集和验证集的COCO格式数据
    with open(train_labels_path, 'w') as f:
        json.dump(train_coco, f, indent=4) 

    with open(val_labels_path, 'w') as f:
        json.dump(val_coco, f, indent=4)  

    # 输出划分结果
    print(f"Train images: {len(train_annotations)}")
    print(f"Val images: {len(val_annotations)}")
