
import os
from PIL import Image
import torch
from torchvision import transforms

class InsectDataset(torch.utils.data.Dataset):
    """昆虫分类数据集。基于YOLO标注，将图像中的每个昆虫目标裁剪出来作为一个样本。"""
    def __init__(self, data_dir, split="train", img_size=224, augmentation=False):
        """
        参数:
            data_dir (str): 数据集根目录，里面应包含 images/<split> 和 labels/<split> 子目录。
            split (str): 数据集划分，"train", "val" 或 "test"。
            img_size (int): 裁剪后的图像将调整到的边长尺寸。
            augmentation (bool): 对训练集是否应用数据增广。
        """
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        self.samples = []  # 存储所有样本的信息: (image_path, class_id, bbox)
        # 图像和标注文件目录
        images_dir = os.path.join(data_dir, "images", split)
        labels_dir = os.path.join(data_dir, "labels", split)
        if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
            raise FileNotFoundError(f"数据集目录 {images_dir} 或 {labels_dir} 不存在")
        # 遍历所有标注文件(.txt)，每行对应一个目标
        for label_file in os.listdir(labels_dir):
            if not label_file.endswith(".txt"):
                continue
            label_path = os.path.join(labels_dir, label_file)
            # 对应的图像文件（尝试jpg和png）
            base_name = os.path.splitext(label_file)[0]
            img_path_jpg = os.path.join(images_dir, base_name + ".jpg")
            img_path_png = os.path.join(images_dir, base_name + ".png")
            if os.path.exists(img_path_jpg):
                img_path = img_path_jpg
            elif os.path.exists(img_path_png):
                img_path = img_path_png
            else:
                continue  # 图像不存在则跳过
            # 读取标注文件的每一行
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    # YOLO格式: class_id, x_center, y_center, width, height (相对坐标)
                    if len(parts) != 5:
                        continue
                    class_id = int(parts[0])
                    x_center, y_center, bw, bh = map(float, parts[1:])
                    # 存储相对坐标，后续在 __getitem__ 中读取图像计算绝对坐标裁剪
                    self.samples.append((img_path, class_id, x_center, y_center, bw, bh))
        # 定义图像变换
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform_list = []
        # 训练集增广：随机水平翻转
        if augmentation:
            transform_list.append(transforms.RandomHorizontalFlip(0.5))
        # 缩放到指定大小
        transform_list.append(transforms.Resize((img_size, img_size)))
        # 将PIL图像转换为Tensor，并归一化
        transform_list.append(transforms.ToTensor())
        transform_list.append(normalize)
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """返回第 idx 个样本 (image_tensor, label_tensor)。"""
        img_path, class_id, x_center, y_center, bw, bh = self.samples[idx]
        # 打开图像并转换为RGB格式
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        # 计算边界框的像素坐标
        x_min = max(0, (x_center - bw/2) * W)
        y_min = max(0, (y_center - bh/2) * H)
        x_max = min(W, (x_center + bw/2) * W)
        y_max = min(H, (y_center + bh/2) * H)
        # 裁剪图像区域并应用变换
        crop = img.crop((x_min, y_min, x_max, y_max))
        if self.transform:
            crop = self.transform(crop)
        # 返回图像Tensor和类别标签
        label = torch.tensor(class_id, dtype=torch.long)
        return crop, label