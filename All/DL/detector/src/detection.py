import os
import yaml
from ultralytics import YOLO

def train_detector(data_dir, class_names, model_name="yolov8n.pt", epochs=50):
    """
    使用Ultralytics YOLOv8训练检测模型。
    参数:
        data_dir (str): 数据集根目录。
        class_names (list): 类别名称列表。
        model_name (str): YOLOv8模型名称或权重路径，例如'yolov8n.pt'。
        epochs (int): 训练轮数。
    返回:
        YOLO 对象: 加载了最佳模型权重的YOLO检测模型。
    """
    # 1️⃣ 优先使用你自己在 data_dir 下写好的 data.yaml
    user_yaml = os.path.join(data_dir, "data.yaml")
    if os.path.exists(user_yaml):
        data_config_path = user_yaml
        print(f"检测到已有配置文件，直接使用: {data_config_path}")
    else:
        # 2️⃣ 如果没有 data.yaml，再回退到原来的自动生成逻辑（agropest_data.yaml）
        data_config_path = os.path.join(data_dir, "agropest_data.yaml")
        if not os.path.exists(data_config_path):
            # 原逻辑：使用 data_dir/images/train 这种结构生成一个配置
            train_images = os.path.join(data_dir, "images", "train")
            val_images = os.path.join(data_dir, "images", "val")
            test_images = os.path.join(data_dir, "images", "test")

            data = {
                "path": data_dir,
                "train": train_images,          # 绝对路径也能用，YOLO 会识别
                "val": val_images,
                # 有 test 就加上
                **({"test": test_images} if os.path.isdir(test_images) else {}),
                "nc": len(class_names),
                "names": {i: name for i, name in enumerate(class_names)},
            }
            with open(data_config_path, "w") as f:
                yaml.dump(data, f)
            print(f"未找到 data.yaml，已自动生成配置文件: {data_config_path}")
        else:
            print(f"使用已存在的配置文件: {data_config_path}")

    # 3️⃣ 加载YOLO模型（如果提供.pt则加载预训练权重）
    model = YOLO(model_name)

    # 4️⃣ 开始训练
    results = model.train(
        data=data_config_path,
        epochs=epochs,
        project="runs",
        name="detect",
        exist_ok=True,
        batch=32
    )

    # 5️⃣ 加载训练所得的最佳模型权重
    best_weights_path = os.path.join("runs", "detect", "weights", "best.pt")
    if os.path.exists(best_weights_path):
        best_model = YOLO(best_weights_path)
        print(f"检测模型训练完成，最佳权重: {best_weights_path}")
    else:
        best_model = model  # 如果未找到best.pt，则返回最后的模型
        print("警告: 未找到 best.pt，返回最后一个 epoch 的模型")

    return best_model