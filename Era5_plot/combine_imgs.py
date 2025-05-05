import os
from PIL import Image, ImageDraw, ImageFont
import re

# **自定义行（数据集）和列（方法）的顺序**
datasets = ["AirQualityUCI", "CaliforniaHousing", "Cifar100", "Era5"]
methods = ["Avg Loss", "S2L", "TTDS"]

# **图片文件夹路径**
image_folder = "./others/"
output_folder = "./others/"
os.makedirs(output_folder, exist_ok=True)

# 读取所有图片
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".png")])

# 创建一个字典来存储按数据集和方法分类的图像
images = {dataset: {} for dataset in datasets}

# 解析文件名并分类
for img_file in image_files:
    match = re.match(r"2_(.*?)_(.*?)_metric_distribution\.png", img_file)
    if match:
        dataset_name, method_name = match.groups()
        dataset_match = next((d for d in datasets if dataset_name.startswith(d)), None)
        method_match = next((m for m in methods if m == method_name), None)

        if dataset_match and method_match:
            img_path = os.path.join(image_folder, img_file)
            images[dataset_match][method_match] = Image.open(img_path)

# **确保所有数据集都有完整的方法图片**
for dataset in datasets:
    for method in methods:
        if method not in images[dataset]:
            raise ValueError(f"缺少 {dataset} 的 {method} 方法的图片！")

# **获取图片尺寸（假设所有图片大小一致）**
img_width, img_height = list(images[datasets[0]].values())[0].size

# **字体设置（默认字体 + 字号 40）**
font_size = 40
try:
    font = ImageFont.truetype("arial.ttf", font_size)  # Windows
except IOError:
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)  # Linux 通常有
    except IOError:
        font = ImageFont.load_default()  # 备用默认字体

# **计算整体图片大小（包括标签）**
padding_x = 80  # 行标题区域宽度（大一些，以适应旋转后的文本）
padding_y = 80  # 列标题区域高度
combined_width = img_width * len(methods) + padding_x
combined_height = img_height * len(datasets) + padding_y

# **创建合并后的图像**
combined_image = Image.new("RGB", (combined_width, combined_height), "white")
draw = ImageDraw.Draw(combined_image)

# ============================================
# **绘制列名称（方法名称）：水平居中且完整显示**
# ============================================
for col_idx, method in enumerate(methods):
    # 计算文本大小
    text_bbox = draw.textbbox((0, 0), method, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # 计算 x 位置：使文本在每个图片区域内水平居中
    x_position = padding_x + col_idx * img_width + (img_width - text_width) // 2
    # 计算 y 位置：使文本在预留的列标题区域内垂直居中
    y_position = (padding_y - text_height) // 2
    draw.text((x_position, y_position), method, fill="black", font=font)

# ============================================
# **绘制行名称（数据集名称，旋转 90°）：解决部分行显示不全的问题**
# ============================================
for row_idx, dataset in enumerate(datasets):
    # 先计算文本的边界框（可能包含负值）
    bbox = draw.textbbox((0, 0), dataset, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # 创建一个透明背景的图像用于绘制文字
    # 注意：由于bbox中可能有负值，所以在绘制时需要进行偏移补偿
    text_img = Image.new("RGBA", (text_width, text_height), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_img)
    # 补偿偏移量：将文字绘制在 (-bbox[0], -bbox[1]) 位置
    text_draw.text((-bbox[0], -bbox[1]), dataset, fill="black", font=font)

    # 旋转文本图像 90°（逆时针）
    rotated_text_img = text_img.rotate(90, expand=True)
    rotated_w, rotated_h = rotated_text_img.size

    # 计算粘贴位置：
    # 在预留的行标题区域（宽度为 padding_x）内水平居中，
    # 同时在每个数据集对应的行内垂直居中
    x_position = (padding_x - rotated_w) // 2
    y_position = padding_y + row_idx * img_height + (img_height - rotated_h) // 2

    # 将旋转后的文本粘贴到主图像中（利用透明通道作为 mask）
    combined_image.paste(rotated_text_img, (x_position, y_position), rotated_text_img)

# ============================================
# **粘贴图像到合成图像中**
# ============================================
for row_idx, dataset in enumerate(datasets):
    for col_idx, method in enumerate(methods):
        img = images[dataset][method]
        combined_image.paste(img, (padding_x + col_idx * img_width, padding_y + row_idx * img_height))

# **保存合并后的图片**
output_path = os.path.join(output_folder, "final_combined_image_rotated_labels.png")
combined_image.save(output_path)

print(f"合并后的图片已保存至: {output_path}")