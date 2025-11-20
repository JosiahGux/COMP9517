# src/dataset.py 关键片段替换
from pathlib import Path
import os
from PIL import Image
import torch
from torchvision import transforms

class InsectDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split="train", img_size=224, augmentation=False):
        super().__init__()
        self.img_size = img_size
        self.samples = []

        # 处理 split 别名
        split = {"valid": "val"}.get(split, split)  # 允许传 valid
        split_candidates = [split]
        if split == "val":
            split_candidates = ["val", "valid"]  # 两个都尝试

        base = Path(data_dir)
        if not base.is_absolute():
            base = (Path(__file__).resolve().parents[1] / base).resolve()

        # 先按  <base>/<split>/images|labels ；再按  <base>/images|labels/<split>
        image_dir_candidates, label_dir_candidates = [], []
        for s in split_candidates:
            image_dir_candidates += [base / s / "images", base / "images" / s]
            label_dir_candidates += [base / s / "labels", base / "labels" / s]

        images_dir = next((d for d in image_dir_candidates if d.is_dir()), None)
        labels_dir = next((d for d in label_dir_candidates if d.is_dir()), None)

        if images_dir is None or labels_dir is None:
            raise FileNotFoundError(f"数据集目录不存在：images候选={image_dir_candidates}  labels候选={label_dir_candidates}")

        # 支持常见后缀
        exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

        # 遍历 labels，匹配同名图片
        for label_file in os.listdir(labels_dir):
            if not label_file.endswith(".txt"):
                continue
            stem = Path(label_file).stem
            img_path = None
            for ext in exts + [e.upper() for e in exts]:
                p = images_dir / f"{stem}{ext}"
                if p.exists():
                    img_path = p
                    break
            if img_path is None:
                continue  # 找不到同名图片就跳过

            with open(labels_dir / label_file, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    parts = s.split()
                    if len(parts) < 5:
                        continue
                    try:
                        cid = int(float(parts[0]))
                        xc, yc, w, h = map(float, parts[1:5])
                    except Exception:
                        continue
                    self.samples.append((str(img_path), cid, xc, yc, w, h))

        if len(self.samples) == 0:
            raise ValueError(
                f"未收集到任何样本，请检查：\n"
                f"- 是否存在 {labels_dir}/*.txt 且内容非空；\n"
                f"- 是否存在 {images_dir}/<同名图片>（后缀 jpg/jpeg/png/bmp/tif）；\n"
                f"- 标注格式是否为 YOLO 五列：class cx cy w h（相对坐标）。"
            )

        # 变换
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        tfs = []
        if augmentation:
            tfs.append(transforms.RandomHorizontalFlip(0.5))
        tfs += [transforms.Resize((img_size, img_size)), transforms.ToTensor(), normalize]
        self.transform = transforms.Compose(tfs)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_id, x_center, y_center, bw, bh = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        x_min = max(0.0, (x_center - bw/2) * W)
        y_min = max(0.0, (y_center - bh/2) * H)
        x_max = min(float(W), (x_center + bw/2) * W)
        y_max = min(float(H), (y_center + bh/2) * H)
        crop = img if (x_max <= x_min or y_max <= y_min) else img.crop((x_min, y_min, x_max, y_max))
        crop = self.transform(crop)
        return crop, torch.tensor(class_id, dtype=torch.long)
