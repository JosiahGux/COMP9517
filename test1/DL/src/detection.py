
import os
import yaml
from ultralytics import YOLO

def train_detector(data_dir, class_names, model_name="yolov8n.pt", epochs=50):
    """
    使用Ultralytics YOLOv8训练检测模型。
    参数:
        data_dir (str): 数据集根目录，包含 images/ 和 labels/ 子目录。
        class_names (list): 类别名称列表。
        model_name (str): YOLOv8模型名称或权重路径，例如'yolov8n.pt'。
        epochs (int): 训练轮数。
    返回:
        YOLO 对象: 加载了最佳模型权重的YOLO检测模型。
    """
    # 准备YOLO数据配置文件
    data_config_path = os.path.join(data_dir, "agropest_data.yaml")
    if not os.path.exists(data_config_path):
        # 列出数据集路径
        train_images = os.path.join(data_dir, "images", "train")
        val_images = os.path.join(data_dir, "images", "val")
        test_images = os.path.join(data_dir, "images", "test")
        data = {
            'path': data_dir,
            'train': os.path.join(data_dir, 'images', 'train'),
            'val': os.path.join(data_dir, 'images', 'val'),
            # 如果有测试集，则加入配置，否则忽略
            **({'test': os.path.join(data_dir, 'images', 'test')} if os.path.isdir(test_images) else {}),
            'names': {i: name for i, name in enumerate(class_names)}
        }
        with open(data_config_path, 'w') as f:
            yaml.dump(data, f)
        print(f"已生成YOLO数据配置文件: {data_config_path}")
    # 加载YOLO模型（如果提供.pt则加载预训练权重，否则.yml则从头开始训练）
    model = YOLO(model_name)
    # 开始训练
    results = model.train(data=data_config_path, epochs=epochs, project="runs", name="detect", exist_ok=True)
    # 加载训练所得的最佳模型权重
    best_weights_path = os.path.join("runs", "detect", "weights", "best.pt")
    if os.path.exists(best_weights_path):
        best_model = YOLO(best_weights_path)
        print(f"检测模型训练完成，最佳权重: {best_weights_path}")
    else:
        best_model = model  # 如果未找到best.pt，则返回最后的模型
    return best_model
