
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import cv2
from PIL import Image
from torchvision import transforms

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
    #model = resnet50(weights=ResNet50_Weights.DEFAULT)
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
        if device.type == "cuda":
            det_model.model.to(device)
    except Exception:
        pass

    os.makedirs(save_dir, exist_ok=True)

    # 与训练时一致的预处理
    IMG_SIZE = 224  # 若训练时用的是别的尺寸，这里改成一致
    preprocess = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    for img_path in image_paths:
        # 1) YOLO 目标检测
        results = det_model(img_path, verbose=False)
        if not results:
            continue
        result = results[0]

        # 2) 读取原图 (BGR)
        image = cv2.imread(img_path)
        if image is None:
            continue
        ih, iw = image.shape[:2]

        # 3) 遍历检测框并做分类
        if getattr(result, "boxes", None) is None or len(result.boxes) == 0:
            # 没有检测到目标，仍然保存原图，便于对照
            save_path = os.path.join(save_dir, os.path.basename(img_path))
            cv2.imwrite(save_path, image)
            continue

        for box in result.boxes:
            # xyxy 绝对坐标 -> int 并裁边
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(iw, x2)), int(min(ih, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            # 裁剪 + 转 RGB
            crop_bgr = image[y1:y2, x1:x2]
            if crop_bgr.size == 0:
                continue
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

            # 转 PIL -> 预处理 -> 张量
            pil_img = Image.fromarray(crop_rgb)
            tensor = preprocess(pil_img).unsqueeze(0).to(device)  # [1,3,H,W]

            # 分类预测
            with torch.no_grad():
                logits = cls_model(tensor)
                probs = torch.softmax(logits, dim=1)[0]
                class_id = int(probs.argmax().item())

            # 类别名：优先 YOLO 的 names（若你的分类标签与检测标签一致）
            class_name = str(class_id)
            if hasattr(det_model, "names"):
                try:
                    if isinstance(det_model.names, dict):
                        class_name = det_model.names.get(class_id, class_name)
                    elif isinstance(det_model.names, (list, tuple)) and 0 <= class_id < len(det_model.names):
                        class_name = det_model.names[class_id]
                except Exception:
                    pass

            # 画框与标签
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
            cv2.putText(image, class_name, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), thickness=2)

        # 4) 保存结果图
        save_path = os.path.join(save_dir, os.path.basename(img_path))
        cv2.imwrite(save_path, image)
