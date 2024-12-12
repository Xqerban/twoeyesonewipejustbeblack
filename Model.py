from ultralytics import YOLO

# 定义训练配置
def train_yolov8(model, train_data, val_data, epochs=10, batch_size=16):
    model.train(
        data=train_data,  # 数据集配置文件路径
        epochs=epochs,
        batch=batch_size,
        imgsz=512,         # 输入图像大小
        device=0,          # 使用的 GPU，设置为0表示使用第一张GPU
        project="runs/train",  # 保存训练结果的文件夹
        name="yolo11m_model",   # 训练任务名称
        exist_ok=True      # 如果文件夹已存在，则覆盖
    )

dataset_config = """
path: data  # 数据集根目录
train: images/train  # 训练集图像路径
val: images/val  # 验证集图像路径
nc: 2  # 类别数量
names: ['no_tampering', 'tampered']  # 类别名称

"""

# 保存数据集配置文件
with open("data.yaml", "w") as f:
    f.write(dataset_config)

if __name__ == '__main__':
    # 确保代码从这里开始
    model = YOLO("yolo11m.pt")
    train_yolov8(model, train_data="data.yaml", val_data="da32ta.yaml", epochs=100, batch_size=16)
