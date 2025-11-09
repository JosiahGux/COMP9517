
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import cv2

def train_classifier(train_dataset, val_dataset, num_classes, device, batch_size=32, epochs=20):
    """
    训练ResNet50分类模型。
    参数:
        train_dataset: 训练集Dataset对象 (由InsectDataset裁剪得到)。
        val_dataset: 验证集Dataset对象。
        num_classes (int): 分类类别数。
        device: 训练设备 (torch.device)。
        batch_size (int): 批次大小。
        epochs (int): 训练轮数。
    返回:
        model: 训练后的分类模型 (已加载最佳权重)。
    """
    # 数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    # 构建ResNet50模型并修改最后一层
    model = models.resnet50(pretrained=True)
    # 冻结特征层 (如果需要可以选择冻结前几层以利用预训练特征)
    # for param in model.parameters():
    #     param.requires_grad = True  # 此处我们选择微调所有参数。如需只训练最后一层，可将其他层requires_grad设为False。
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    best_acc = 0.0
    os.makedirs("runs/classifier", exist_ok=True)
    best_model_path = os.path.join("runs", "classifier", "best.pth")
    # 开始训练循环
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        total_train = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * images.size(0)
            total_train += labels.size(0)
        train_loss = total_train_loss / total_train if total_train > 0 else 0.0
        # 验证阶段
        model.eval()
        total_val = 0
        correct_val = 0
        total_val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item() * images.size(0)
                # 计算预测准确数
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
        val_loss = total_val_loss / total_val if total_val > 0 else 0.0
        val_acc = correct_val / total_val if total_val > 0 else 0.0
        print(f"Epoch [{epoch+1}/{epochs}]: "
              f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        # 保存当前最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
    # 训练结束，加载最佳模型权重
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"分类模型训练完成，最佳验证准确率={best_acc:.4f}，权重已保存: {best_model_path}")
    else:
        print("未找到最佳模型权重，可能训练未正常完成。")
    return model

def evaluate_classifier(model, dataset, device):
    """
    评估分类模型在给定数据集上的准确率。
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    model.eval().to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total if total > 0 else 0.0
    split = getattr(dataset, "split", "data")
    print(f"{split} 集合分类准确率: {acc*100:.2f}% ({correct}/{total})")
    return acc

def load_classifier_model(num_classes, weights_path, device):
    """
    加载训练好的ResNet50分类模型。
    """
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device).eval()
    return model

def inference_on_images(det_model, cls_model, image_paths, device, save_dir="runs/inference"):
    """
    对给定的图像列表执行检测和分类，并将带注释的结果图像保存。
    参数:
        det_model: 训练好的YOLO检测模型 (ultralytics.YOLO 对象)。
        cls_model: 训练好的分类模型 (torch.nn.Module)。
        image_paths (list): 图像路径列表。
        device: 设备 (torch.device)。
        save_dir (str): 保存结果的目录。
    """
    cls_model = cls_model.to(device).eval()
    # 确保检测模型使用GPU（若可用）
    try:
        if device.type == 'cuda':
            det_model.model.to(device)
    except Exception:
        pass
    os.makedirs(save_dir, exist_ok=True)
    for img_path in image_paths:
        # 使用YOLO模型进行目标检测
        results = det_model(img_path, verbose=False)  # 获得检测结果
        if len(results) == 0:
            continue
        result = results[0]
        # 读取原始图像用于绘图（OpenCV以BGR格式读入）
        image = cv2.imread(img_path)
        ih, iw, _ = image.shape
        # 遍历每个检测到的目标
        for box in result.boxes:
            # 提取边界框坐标和置信度（xyxy为绝对坐标）
            x1, y1, x2, y2 = box.xyxy[0]  # tensor格式
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # 裁剪目标区域并进行分类预测
            crop_img = image[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            # 缩放裁剪图像到分类输入大小
            crop_resized = cv2.resize(crop_rgb, (224, 224))
            # 转换为Tensor并归一化（与训练时相同）
            tensor = torch.from_numpy(crop_resized.transpose(2, 0, 1)).float().div(255.0).to(device)
            # 按ImageNet均值和标准差规范化
            tensor = torch.nn.functional.normalize(tensor,
                        mean=torch.tensor([0.485, 0.456, 0.406], device=device),
                        std=torch.tensor([0.229, 0.224, 0.225], device=device))
            tensor = tensor.unsqueeze(0)  # 添加batch维度
            with torch.no_grad():
                outputs = cls_model(tensor)
                _, pred_cls = torch.max(outputs, 1)
                class_id = int(pred_cls.cpu().item())
            # 获取类别名称和绘制结果
            class_name = str(class_id)
            if hasattr(det_model, 'names'):
                # YOLO模型自带类别名
                class_name = det_model.names.get(class_id, class_name)
            # 绘制矩形框和类别标签
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
            cv2.putText(image, class_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), thickness=2)
        # 保存带注释的图像
        save_path = os.path.join(save_dir, os.path.basename(img_path))
        cv2.imwrite(save_path, image)