import os
import argparse
import glob
import torch
from ultralytics import YOLO

# 导入自定义模块
from src import detection, classification, dataset

# 配置：数据集路径和类别名称
DATA_DIR = "archive"  # 数据集根目录，包含 images/ 和 labels/ 子目录
CLASS_NAMES = ["Ants", "Bees", "Beetles", "Caterpillars", "Earthworms",
               "Earwigs", "Grasshoppers", "Moths", "Slugs", "Snails", "Wasps", "Weevils"]  # 12类害虫名称
NUM_CLASSES = len(CLASS_NAMES)
# 训练参数配置
EPOCHS_DETECTOR = 50       # 检测器训练轮数
EPOCHS_CLASSIFIER = 50     # 分类器训练轮数
BATCH_SIZE = 32            # 分类模型训练的批次大小
IMG_SIZE = 224             # 分类器输入图像大小
YOLO_MODEL = "yolov8n.pt"  # YOLOv8预训练模型权重 (可选用yolov8s.pt等)
# 路径配置
DETECT_RUN_NAME = "detect"  # YOLO训练输出文件夹名称
DETECT_RUN_DIR = os.path.join("runs", DETECT_RUN_NAME)
DET_BEST_WEIGHTS = os.path.join(DETECT_RUN_DIR, "weights", "best.pt")
CLS_BEST_WEIGHTS = os.path.join("runs", "classifier", "best.pth")

# 创建输出目录
os.makedirs("runs", exist_ok=True)

# 解析命令行参数
parser = argparse.ArgumentParser(description="Insect Detection and Classification Pipeline")
parser.add_argument("--task", choices=['train_detector', 'train_classifier', 'train_all', 'inference', 'evaluate'],
                    default='train_all', help="选择要执行的任务: train_detector/train_classifier/train_all/inference/evaluate")
parser.add_argument("--image", type=str, help="在 inference 模式下指定单张待预测的图像路径")
parser.add_argument("--num_images", type=int, default=10, help="inference 模式下随机抽取的图片数量")
args = parser.parse_args()

# 自动选择设备：GPU优先，否则CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 根据任务执行对应流程
if args.task == 'train_detector':
    # 训练YOLOv8检测模型
    det_model = detection.train_detector(data_dir=DATA_DIR, class_names=CLASS_NAMES,
                                         model_name=YOLO_MODEL, epochs=EPOCHS_DETECTOR)
elif args.task == 'train_classifier':
    # 构建训练和验证数据集（使用标注裁剪昆虫目标）
    train_dataset = dataset.InsectDataset(data_dir=DATA_DIR, split="train", img_size=IMG_SIZE, augmentation=True)
    val_dataset = dataset.InsectDataset(data_dir=DATA_DIR, split="val", img_size=IMG_SIZE, augmentation=False)
    # 训练ResNet50分类模型
    cls_model = classification.train_classifier(train_dataset, val_dataset,
                                                num_classes=NUM_CLASSES, device=device,
                                                batch_size=BATCH_SIZE, epochs=EPOCHS_CLASSIFIER)
elif args.task == 'train_all':
    # 依次训练检测模型和分类模型，然后评估性能并示例推理
    det_model = detection.train_detector(data_dir=DATA_DIR, class_names=CLASS_NAMES,
                                         model_name=YOLO_MODEL, epochs=EPOCHS_DETECTOR)
    # 准备分类训练数据（利用检测标注进行裁剪）
    train_dataset = dataset.InsectDataset(data_dir=DATA_DIR, split="train", img_size=IMG_SIZE, augmentation=True)
    val_dataset = dataset.InsectDataset(data_dir=DATA_DIR, split="val", img_size=IMG_SIZE, augmentation=False)
    cls_model = classification.train_classifier(train_dataset, val_dataset,
                                                num_classes=NUM_CLASSES, device=device,
                                                batch_size=BATCH_SIZE, epochs=EPOCHS_CLASSIFIER)
    # 评估模型性能
    print("评估检测模型（YOLOv8）在验证集上的表现...")
    det_model.val()  # 使用YOLO内置方法计算mAP等指标
    # 评估分类模型在测试集或验证集上的准确率
    eval_split = "test" if os.path.isdir(os.path.join(DATA_DIR, "images", "test")) else "val"
    eval_dataset = dataset.InsectDataset(data_dir=DATA_DIR, split=eval_split, img_size=IMG_SIZE, augmentation=False)
    classification.evaluate_classifier(cls_model, eval_dataset, device=device)
    # 输出若干预测结果示例图像
    print("输出推理示例结果至 runs/inference/ 文件夹...")
    run_images = []
    infer_dir = os.path.join(DATA_DIR, "images", eval_split)
    # 获取待推理的图像路径（随机选取5张）
    all_images = sorted(glob.glob(os.path.join(infer_dir, "*.*")))
    for img_path in all_images[:5]:
        run_images.append(img_path)
    if run_images:
        os.makedirs("runs/inference", exist_ok=True)
        classification.inference_on_images(det_model, cls_model, run_images, device=device, save_dir="runs/inference")
    print("完整流程(train_all)执行完毕。")
elif args.task == 'inference':
    # 推理：加载已训练的模型权重并对指定图像或示例图像进行预测
    if not os.path.exists(DET_BEST_WEIGHTS) or not os.path.exists(CLS_BEST_WEIGHTS):
        print("未找到已训练模型权重，请先运行训练流程。")
        exit(1)

    # 加载检测模型和分类模型
    det_model = YOLO(DET_BEST_WEIGHTS)
    cls_model = classification.load_classifier_model(num_classes=NUM_CLASSES, weights_path=CLS_BEST_WEIGHTS, device=device)

    # 若指定单张图，直接用该图
    if args.image:
        images = [args.image]
    else:
        import random
        random.seed(42)  # 如需可复现抽样

        def find_split_images(base, splits=("test", "val", "valid", "train")):
            # 优先找 archive/<split>/images，其次 archive/images/<split>
            for s in splits:
                p1 = os.path.join(base, s, "images")      # 你的结构：archive/test/images
                p2 = os.path.join(base, "images", s)      # 另一种：archive/images/test
                if os.path.isdir(p1):
                    return p1
                if os.path.isdir(p2):
                    return p2
            return None

        sample_dir = find_split_images(DATA_DIR)
        if not sample_dir:
            raise FileNotFoundError("未找到可用于推理的 images 目录（test/val/valid/train）。请检查数据结构或传入 --image。")

        # 收集所有图片（常见后缀）
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.TIF", "*.TIFF")
        all_images = []
        for pat in exts:
            all_images.extend(glob.glob(os.path.join(sample_dir, pat)))
        all_images = sorted(all_images)

        if not all_images:
            raise FileNotFoundError(f"目录中未找到图片文件: {sample_dir}")

        # 随机抽取指定数量
        k = min(args.num_images, len(all_images))
        images = random.sample(all_images, k)

    # 执行推理并保存结果图像
    os.makedirs("runs/inference", exist_ok=True)
    classification.inference_on_images(det_model, cls_model, images, device=device, save_dir="runs/inference")
    print(f"推理完成，结果保存在 runs/inference/ 目录。抽取了 {len(images)} 张。")

elif args.task == 'evaluate':
    # 评估：计算检测模型的mAP和分类模型的准确率
    if not os.path.exists(DET_BEST_WEIGHTS) or not os.path.exists(CLS_BEST_WEIGHTS):
        print("未找到已训练模型权重，请先训练模型。")
        exit(1)
    det_model = YOLO(DET_BEST_WEIGHTS)
    det_model.val(split='test')
    # 使用YOLO对验证集评估（若存在测试集，可修改data/agropest_data.yaml的路径并指定split）
    print("评估YOLOv8模型在验证集上的检测性能...")
    #det_model.val()
    # 评估分类模型
    #eval_split = "test" if os.path.isdir(os.path.join(DATA_DIR, "images", "test")) else "val"
    #eval_dataset = dataset.InsectDataset(data_dir=DATA_DIR, split=eval_split, img_size=IMG_SIZE, augmentation=False)
    def has_split_images(base, s):
        # 你的结构：archive/<split>/images
        p1 = os.path.join(base, s, "images")
        # 另一种：archive/images/<split>
        p2 = os.path.join(base, "images", s)
        return os.path.isdir(p1) or os.path.isdir(p2)

    if has_split_images(DATA_DIR, "test"):
        eval_split = "test"
    elif has_split_images(DATA_DIR, "val") or has_split_images(DATA_DIR, "valid"):
        eval_split = "val"  # InsectDataset 兼容 val/valid
    else:
        eval_split = "train"  # 实在没有就退回 train

    eval_dataset = dataset.InsectDataset(data_dir=DATA_DIR, split=eval_split, img_size=IMG_SIZE, augmentation=False)

    cls_model = classification.load_classifier_model(num_classes=NUM_CLASSES, weights_path=CLS_BEST_WEIGHTS, device=device)
    classification.evaluate_classifier(cls_model, eval_dataset, device=device)