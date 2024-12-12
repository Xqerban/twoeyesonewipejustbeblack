from PIL import Image
import os

# 输入和输出路径
input_folder = 'data/image/val'  # 输入文件夹，包含 PNG 和 JPEG 图片
output_folder = 'data/image/val1'  # 输出文件夹，保存 JPG 图片

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 批量处理文件
for image_name in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_name)

    # 只处理 PNG 和 JPEG 格式的文件
    if image_name.lower().endswith(('.png', '.jpeg', '.jpg')):
        # 打开图片
        with Image.open(image_path) as img:
            # 转换为 RGB 模式，以便保存为 JPG
            img = img.convert('RGB')

            # 生成新的文件名
            output_path = os.path.join(output_folder, os.path.splitext(image_name)[0] + '.jpg')

            # 保存为 JPG 格式
            img.save(output_path, 'JPEG')

            print(f"已转换：{image_name} -> {os.path.basename(output_path)}")

print("所有图片已转换完成。")
