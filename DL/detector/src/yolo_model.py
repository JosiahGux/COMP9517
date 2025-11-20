import os
from typing import List, Optional, Union, Dict, Any

from ultralytics import YOLO


class YOLOModel:
    """
    YOLOModel 是对 Ultralytics YOLOv8 的一个高层封装。

    设计目标：
      1. 把 YOLO 的训练 / 评估 / 推理流程统一成一个类，便于在 main.py 中调用；
      2. 允许方便地切换不同规模的 YOLO 模型（yolov8n/s/m...）；
      3. 便于在报告中展示“我们自己封装了检测器模块”，而不是简单调用 YOLO.train。

    使用方式示例（在其他文件中）：
        from yolo_model import YOLOModel

        yolo = YOLOModel(
            model_name="yolov8m.pt",
            data_config="archive/data.yaml",
            project="runs",
            exp_name="detect",
            batch_size=32,
            img_size=640,
            device="cuda",
        )

        # 训练
        yolo.train(epochs=50)

        # 在验证集上评估
        yolo.evaluate(split="val")

        # 在测试集上评估
        yolo.evaluate(split="test")

        # 推理
        yolo.predict(source="archive/test/images", conf=0.25)
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        data_config: Optional[str] = None,
        project: str = "runs",
        exp_name: str = "detect",
        batch_size: int = 16,
        img_size: int = 640,
        device: str = "cuda",
    ) -> None:
        """
        初始化 YOLOModel。

        参数:
            model_name: YOLO 模型名称或权重路径。
                        - 'yolov8n.pt' / 'yolov8s.pt' / 'yolov8m.pt'：使用官方预训练权重
                        - 'runs/detect/weights/best.pt'：加载自己训练好的权重
            data_config: YOLO 数据集配置文件路径（data.yaml）。
            project: 实验输出的根目录（例如 'runs'）。
            exp_name: 当前实验名称（例如 'detect'，最终路径类似 runs/detect/）。
            batch_size: 训练时的 batch 大小。
            img_size: 输入图片尺寸（方形边长）。
            device: 训练 / 推理设备，'cuda' 或 'cpu'。
        """
        self.model_name = model_name
        self.data_config = data_config
        self.project = project
        self.exp_name = exp_name
        self.batch_size = batch_size
        self.img_size = img_size
        self.device = device

        # 结果目录下常见的权重命名
        self.exp_dir = os.path.join(self.project, self.exp_name)
        self.best_weights_path = os.path.join(self.exp_dir, "weights", "best.pt")
        self.last_weights_path = os.path.join(self.exp_dir, "weights", "last.pt")

        # 1. 先加载基础模型（可能是预训练权重，也可能是结构配置）
        print(f"[YOLOModel] 初始化模型: {self.model_name}")
        self.model = YOLO(self.model_name)

        # 2. 如果已经存在 best.pt，优先加载 best（方便断点续训 / 复现最优结果）
        if os.path.exists(self.best_weights_path):
            print(f"[YOLOModel] 检测到已有最优权重，优先加载: {self.best_weights_path}")
            self.model = YOLO(self.best_weights_path)

    # ------------------------------------------------------------------
    # 一些工具方法（辅助）
    # ------------------------------------------------------------------
    def has_trained_weights(self) -> bool:
        """检查是否已经存在训练好的 best 权重。"""
        return os.path.exists(self.best_weights_path)

    def load_best_weights(self) -> None:
        """手动加载 best 权重（如果存在的话）。"""
        if self.has_trained_weights():
            print(f"[YOLOModel] 手动加载最优权重: {self.best_weights_path}")
            self.model = YOLO(self.best_weights_path)
        else:
            print("[YOLOModel] 未找到最优权重 best.pt，保持当前模型不变。")

    # ------------------------------------------------------------------
    # 训练接口
    # ------------------------------------------------------------------
    def train(
        self,
        epochs: int = 50,
        lr0: float = 0.01,
        lrf: float = 0.01,
        optimizer: str = "auto",
        weight_decay: float = 0.0005,
        freeze: Union[int, List[int]] = 0,
        verbose: bool = True,
        extra_train_args: Optional[Dict[str, Any]] = None,
    ):
        """
        使用指定的数据配置 data.yaml 训练 YOLO 模型。

        参数:
            epochs: 训练轮数。
            lr0: 初始学习率。
            lrf: 最终学习率与 lr0 的比例（cosine 下降到 lr0 * lrf）。
            optimizer: 优化器类型，'auto' / 'SGD' / 'Adam' / 'AdamW' 等。
            weight_decay: 权重衰减系数（L2 正则化）。
            freeze: 冻结前多少层，或一个列表指定需要冻结的层（通常 0 表示不冻结）。
            verbose: 是否打印训练的详细日志。
            extra_train_args: 额外透传给 YOLO.train() 的参数字典，例如：
                              {'mosaic': 0.7, 'mixup': 0.1}
        """
        if self.data_config is None:
            raise ValueError("未指定 data_config（data.yaml），无法训练 YOLO 模型。")

        if extra_train_args is None:
            extra_train_args = {}

        if verbose:
            print("=" * 60)
            print("[YOLOModel] 开始训练 YOLO 模型")
            print(f"  - 数据配置文件: {self.data_config}")
            print(f"  - 实验目录: {self.exp_dir}")
            print(f"  - 模型: {self.model_name}")
            print(f"  - 设备: {self.device}")
            print(f"  - epochs: {epochs}, batch: {self.batch_size}, imgsz: {self.img_size}")
            print(f"  - optimizer: {optimizer}, lr0: {lr0}, lrf: {lrf}, weight_decay: {weight_decay}")
            print(f"  - freeze: {freeze}")
            if extra_train_args:
                print(f"  - 额外训练参数: {extra_train_args}")
            print("=" * 60)

        results = self.model.train(
            data=self.data_config,
            epochs=epochs,
            imgsz=self.img_size,
            batch=self.batch_size,
            device=self.device,
            project=self.project,
            name=self.exp_name,
            exist_ok=True,
            optimizer=optimizer,
            lr0=lr0,
            lrf=lrf,
            weight_decay=weight_decay,
            freeze=freeze,
            verbose=verbose,
            **extra_train_args,
        )

        # 训练结束后，若存在 best.pt，则切换为最优权重
        if os.path.exists(self.best_weights_path):
            print(f"[YOLOModel] 训练完成，加载最优权重: {self.best_weights_path}")
            self.model = YOLO(self.best_weights_path)
        else:
            print("[YOLOModel] 训练完成，但未找到 best.pt，将继续使用当前模型权重。")

        return results

    # ------------------------------------------------------------------
    # 评估接口
    # ------------------------------------------------------------------
    def evaluate(
        self,
        split: str = "val",
        conf: float = 0.001,
        iou: float = 0.6,
        save_json: bool = False,
        plots: bool = True,
    ):
        """
        在验证集或测试集上评估检测性能。

        参数:
            split: 评估使用的数据划分，通常为 'val' 或 'test'。
                   需要在 data.yaml 里对应配置 val/test 路径。
            conf: 推理时的置信度阈值（用于 NMS 前过滤）。
            iou: NMS 的 IoU 阈值。
            save_json: 是否保存 COCO 格式的评估结果（通常用于 COCO 官方 API）。
            plots: 是否生成 PR 曲线、F1 曲线、混淆矩阵等可视化图。
        """
        if self.data_config is None:
            raise ValueError("未指定 data_config（data.yaml），无法评估 YOLO 模型。")

        print(f"[YOLOModel] 在 {split} 集上评估 (conf={conf}, iou={iou}) ...")

        results = self.model.val(
            data=self.data_config,
            split=split,
            imgsz=self.img_size,
            conf=conf,
            iou=iou,
            device=self.device,
            project=self.project,
            name=f"{self.exp_name}_val_{split}",
            exist_ok=True,
            save_json=save_json,
            plots=plots,
        )

        # 一些常用指标，可以在报告或者日志里使用
        try:
            map50_95 = results.box.map      # mAP@0.5:0.95
            map50 = results.box.map50      # mAP@0.5
            print(f"[YOLOModel] 评估完成: mAP50={map50:.3f}, mAP50-95={map50_95:.3f}")
        except Exception:
            print("[YOLOModel] 无法从结果中解析 mAP 指标（可能是版本差异），请手动检查 results。")

        return results

    # ------------------------------------------------------------------
    # 推理接口
    # ------------------------------------------------------------------
    def predict(
        self,
        source: Union[str, List[str]],
        save_dir: Optional[str] = None,
        conf: float = 0.25,
        iou: float = 0.45,
        show: bool = False,
        save_txt: bool = False,
        save_conf: bool = False,
    ):
        """
        对单张图片、文件夹或图片路径列表进行推理，并将结果保存到指定目录。

        参数:
            source: 图片路径、视频路径、文件夹路径或路径列表。
                    例如：
                        - 'archive/test/images/0001.jpg'
                        - 'archive/test/images'
                        - ['img1.jpg', 'img2.jpg']
            save_dir: 推理结果保存目录（若为 None，则自动放在 runs/exp_name_predict 下）。
            conf: 置信度阈值。
            iou: NMS 阈值。
            show: 是否在本地弹出窗口显示结果（服务器环境通常为 False）。
            save_txt: 是否保存 YOLO 格式的 txt 检测结果。
            save_conf: 是否在 txt 中额外保存置信度。
        """
        if save_dir is None:
            save_dir = os.path.join(self.project, f"{self.exp_name}_predict")

        os.makedirs(save_dir, exist_ok=True)

        print(f"[YOLOModel] 开始推理")
        print(f"  - source: {source}")
        print(f"  - 结果保存目录: {save_dir}")

        results = self.model.predict(
            source=source,
            imgsz=self.img_size,
            conf=conf,
            iou=iou,
            device=self.device,
            project=self.project,
            name=os.path.basename(save_dir),
            save=True,          # 保存带可视化结果的图片 / 视频
            save_txt=save_txt,  # 若 True，则保存 .txt 结果（边框 + 类别）
            save_conf=save_conf,
            show=show,
            exist_ok=True,
        )

        return results