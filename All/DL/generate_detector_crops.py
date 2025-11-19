import os
import cv2
from ultralytics import YOLO
from tqdm import tqdm

def generate_detector_crops(
    det_weights,
    data_root,
    out_root,
    splits=("train", "valid", "test"),
    img_subdir="images",
    conf=0.25,
    iou=0.45,
    img_size=640,
    class_names=None,
):
    """
    使用训练好的 YOLO 检测器，对 data_root 下的原始图片进行检测，
    将每个检测到的目标裁剪出来，按类别保存到 out_root 目录下。

    参数:
        det_weights (str): YOLO 检测器权重路径，比如 'runs/detect/weights/best.pt'
        data_root (str): 原始数据根目录，结构类似：
                          data_root/
                            train/images
                            valid/images
                            test/images
        out_root (str): 裁剪后小图的输出目录，自动创建，结构会变成：
                          out_root/
                            train/Ants/*.jpg
                            train/Bees/*.jpg
                            ...
                            valid/Ants/*.jpg
                            ...
        splits (tuple): 要处理的数据划分，默认 ("train", "valid", "test")
        img_subdir (str): 每个 split 下存放图片的子目录名，通常是 "images"
        conf (float): YOLO 推理置信度阈值
        iou (float): YOLO NMS 的 IoU 阈值
        img_size (int): YOLO 推理的输入尺寸
        class_names (list or None): 类别名称列表，如果为 None，则使用 YOLO 模型自带的 names
    """
    # 1. 加载检测器
    model = YOLO(det_weights)
    if class_names is None:
        # 使用模型内部的类别名
        names = model.names  # dict: id -> name
    else:
        # 用我们自己传入的类别名
        names = {i: n for i, n in enumerate(class_names)}

    # 2. 遍历 train / valid / test
    for split in splits:
        img_dir = os.path.join(data_root, split, img_subdir)
        if not os.path.isdir(img_dir):
            print(f"[WARN] 找不到目录: {img_dir}，跳过 {split}")
            continue

        # 输出目录：out_root/split/
        out_split_dir = os.path.join(out_root, split)
        os.makedirs(out_split_dir, exist_ok=True)

        img_files = [f for f in os.listdir(img_dir)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        print(f"[INFO] 处理 {split} 集: {len(img_files)} 张图片")
        for fname in tqdm(img_files):
            img_path = os.path.join(img_dir, fname)
            # 用 OpenCV 读原图（BGR）
            image = cv2.imread(img_path)
            if image is None:
                continue
            h, w, _ = image.shape

            # 3. 用 YOLO 检测该图
            results = model.predict(
                source=image,
                imgsz=img_size,
                conf=conf,
                iou=iou,
                verbose=False
            )

            if len(results) == 0:
                continue
            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                continue

            # 4. 对每个检测框裁剪并保存
            for det_id, box in enumerate(boxes):
                cls_id = int(box.cls[0].item())
                x1, y1, x2, y2 = box.xyxy[0]  # tensor
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                # 边界保护
                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h))
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = image[y1:y2, x1:x2]  # BGR

                class_name = names.get(cls_id, f"class_{cls_id}")
                # 按类别建文件夹：out_root/split/class_name/
                class_dir = os.path.join(out_split_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)

                # 保存文件名：原图名_检测序号.jpg
                base, ext = os.path.splitext(fname)
                save_name = f"{base}_det{det_id}{ext}"
                save_path = os.path.join(class_dir, save_name)

                cv2.imwrite(save_path, crop)

    print(f"[DONE] 所有裁剪图片已保存到: {out_root}")