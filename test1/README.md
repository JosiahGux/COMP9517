


**main.py 说明**：
- 首先设置了数据路径和类别列表，以及训练的参数（可根据需要修改模型类型、训练轮数等）。
- 使用 `argparse` 模块定义了 `--task` 参数来控制运行阶段：
  - `train_detector`：仅训练YOLOv8检测器；
  - `train_classifier`：仅训练ResNet50分类器（需确保有检测标注文件来裁剪训练图像）；
  - `train_all`：顺序执行检测器训练->分类器训练->评估->输出示例结果；
  - `inference`：使用已有模型对指定图像或默认样例图像进行推理；
  - `evaluate`：评估模型在验证/测试集上的性能。
- `train_all` 为默认选项，实现了一键式全流程：训练完成后会自动评估模型，并将若干预测结果绘制后保存在 `runs/inference/` 下。
- 设备选择上，代码使用 `torch.cuda.is_available()` 自动检测GPU，如有则使用GPU否则使用CPU [oai_citation:4‡blog.roboflow.com](https://blog.roboflow.com/how-to-train-yolov8-on-a-custom-dataset/#:~:text=Following%20the%20trend%20set%20by,js%20or%20CoreML)。
- 对于推理和评估模式，会先检查预训练权重文件是否存在，否则提示先训练。推理阶段可通过 `--image` 指定单个图像路径，未指定则程序会从测试或验证集中挑选数张图片进行示例。

### 数据集处理模块 (src/dataset.py)

该模块负责将原始数据集格式处理成模型可用的形式。对于检测器YOLOv8，其需要一个数据配置文件和按照规定组织的文件夹；在我们的实现中，这部分由 `detection.py` 动态生成（见下节）。对于分类器，我们需要基于YOLO标注，将每个昆虫目标裁剪为单独的小图像用于训练分类模型 [oai_citation:5‡reddit.com](https://www.reddit.com/r/learnmachinelearning/comments/144erpn/has_anyone_used_object_detection_and/#:~:text=Yes%20you%20can,Worked%20great)。`InsectDataset` 类实现了这一功能：它读取指定 split (train/val/test) 下的所有标注文件，每条标注对应一个昆虫目标裁剪样本。初始化时可以选择是否进行数据增广（如随机翻转等）。裁剪后的图像会被调整到统一大小（224x224），并转换为Tensor，同时归一化到ResNet预训练的图像分布。下面是 `dataset.py` 的实现：


**dataset.py 说明**：
- `InsectDataset` 会在初始化时遍历指定`split`的所有标注文件，将每个标注项转换成一个样本条目。每个样本由原始图像路径、类别ID以及标注的相对边界框坐标组成。实际取样时（`__getitem__`），再根据图像尺寸计算出边界框的像素坐标并裁剪。
- 裁剪后的小图会调整为 `img_size` 大小（默认224x224），并转换为Tensor。对于训练集（augmentation=True），还会随机水平翻转，以扩充数据。最后使用ImageNet预训练的均值和标准差进行归一化，这与ResNet50预训练时的数据分布一致，有助于迁移学习效果 [oai_citation:6‡brsoff.github.io](https://brsoff.github.io/tutorials/beginner/finetuning_torchvision_models_tutorial.html#:~:text=,Run%20the%20training%20step)。
- 该Dataset为分类模型提供了逐虫目标的图像和标签。注意，这里直接使用标注的真实边界框来生成分类训练数据，而**不依赖**检测模型的预测输出，从而确保分类器训练数据的准确性。

### 检测模型模块 (src/detection.py)

该模块封装了YOLOv8模型的训练流程。使用 Ultralytics 提供的YOLO Python接口，可以方便地加载预训练模型并在自定义数据集上微调训练 [oai_citation:7‡blog.roboflow.com](https://blog.roboflow.com/how-to-train-yolov8-on-a-custom-dataset/#:~:text=The%20developers%20of%20YOLOv8%20decided,it%20is%20a%20fantastic%20decision)。`train_detector` 函数会生成YOLO需要的数据配置文件（data.yaml），然后调用 Ultralytics YOLO 接口开始训练。训练完成后，函数返回加载了最佳权重的模型供后续使用。Ultralytics YOLOv8 模型在PyTorch上实现，能自行检测使用CPU或GPU [oai_citation:8‡blog.roboflow.com](https://blog.roboflow.com/how-to-train-yolov8-on-a-custom-dataset/#:~:text=Following%20the%20trend%20set%20by,js%20or%20CoreML)。下面是 `detection.py` 的实现：

**detection.py 说明**：
- `train_detector` 首先检查并创建YOLO的数据配置文件 `agropest_data.yaml`，其中包含训练、验证（和可选测试）集的路径以及类别名称映射。Ultralytics YOLO要求这样的配置文件来了解数据集结构。我们这里假设数据已按 `images/train, images/val, labels/train, labels/val` 组织。如未拆分验证集，可将部分训练数据手动移动到`val`目录或在yaml中指向相同目录。
- 然后，利用 Ultralytics YOLO 的Python接口加载模型。如果 `model_name` 为 `.pt` 文件（比如 `'yolov8n.pt'`），则会加载COCO预训练的YOLOv8nano模型权重用于微调；也可以设置为 `.yaml` 模型配置文件以从零训练。
- 调用 `model.train(...)` 开始训练 [oai_citation:9‡docs.ultralytics.com](https://docs.ultralytics.com/usage/python/#:~:text=,yolo11n.yaml)。这里指定 `project="runs"`, `name="detect"`，训练日志和模型权重将保存在 `runs/detect/` 目录下（Ultralytics默认会将最佳模型保存为 `best.pt`）。训练过程中，YOLOv8会自动利用GPU加速（若可用）或CPU运行 [oai_citation:10‡blog.roboflow.com](https://blog.roboflow.com/how-to-train-yolov8-on-a-custom-dataset/#:~:text=Following%20the%20trend%20set%20by,js%20or%20CoreML)。
- 训练完成后，函数尝试加载 `best.pt` 权重返回一个新的YOLO模型实例。如因配置未保存最佳权重，则退而使用最后的模型。返回的 `best_model` 可用于后续的验证、推理阶段。
- **注意**：Ultralytics库在训练期间会输出各epoch的损失和mAP等指标，并在`runs/detect`下保存模型和结果图表等。可在README中进一步说明。

### 分类模型模块 (src/classification.py)

该模块实现ResNet50分类模型的构建、训练、评估和推理等功能。我们采用Torchvision提供的预训练ResNet50模型，并修改其最后的全连接层输出维度为12以适配我们的类别数 [oai_citation:11‡brsoff.github.io](https://brsoff.github.io/tutorials/beginner/finetuning_torchvision_models_tutorial.html#:~:text=,want%20to%20update%20during%20training)。训练时，使用前面准备的 `InsectDataset` 提供的裁剪虫体图像进行微调 [oai_citation:12‡reddit.com](https://www.reddit.com/r/learnmachinelearning/comments/144erpn/has_anyone_used_object_detection_and/#:~:text=Yes%20you%20can,Worked%20great)。`train_classifier` 函数负责训练并保存最佳模型权重，`evaluate_classifier` 计算分类准确率，`load_classifier_model` 用于加载已训练的分类模型权重，`inference_on_images` 则对给定图像列表执行目标检测+分类并将结果可视化。下面是 `classification.py` 的实现：


**classification.py 说明**：  
- `train_classifier`：使用训练集和验证集的数据加载器，训练ResNet50模型。我们利用ImageNet预训练的ResNet50初始化模型，并将最后的全连接层输出维度修改为 `num_classes=12` [oai_citation:13‡brsoff.github.io](https://brsoff.github.io/tutorials/beginner/finetuning_torchvision_models_tutorial.html#:~:text=,want%20to%20update%20during%20training)。可以选择冻结部分层以进行特征提取，但此处默认微调所有参数。优化器使用Adam，学习率1e-4，可根据需要调整。每个epoch结束后在验证集计算损失和准确率，并打印日志。若当前模型在验证集上的准确率高于历史最佳，则保存模型参数到 `runs/classifier/best.pth`。训练完成后，加载最佳权重返回模型实例用于后续使用。  
- `evaluate_classifier`：对给定的数据集（验证集或测试集）计算分类准确率，打印正确率百分比及正确/总样本数。  
- `load_classifier_model`：用于在推理或评估时载入训练好的ResNet50模型权重，返回模型（已设置为eval模式）。  
- `inference_on_images`：实现了完整的推理流程。它对每张输入图像，先用YOLOv8检测模型获得预测框，然后对每个检测出的框裁剪出区域，通过分类模型判断类别。最后使用OpenCV在原图上绘制边界框和类别标签，并将结果图像保存到指定文件夹。这个过程中，我们确保分类模型使用与训练时相同的图像预处理（包括缩放和归一化），以保证预测准确性。绘制结果时优先使用YOLO模型自带的类别名称映射（`det_model.names`）来将类别ID转换为人类可读的名称。如图像中有多个目标，会全部标注。  

### README.md 和使用说明

（**README.md 概要**）：

项目简介：  
本项目通过两个深度学习模型实现对农作物害虫的检测与分类：YOLOv8负责检测昆虫的位置，ResNet50对检测出的昆虫进行品种分类。项目使用PyTorch框架，支持GPU加速 [oai_citation:14‡blog.roboflow.com](https://blog.roboflow.com/how-to-train-yolov8-on-a-custom-dataset/#:~:text=Following%20the%20trend%20set%20by,js%20or%20CoreML)，并在公开数据集AgroPest-12上训练和验证。该数据集包含蚂蚁、蜜蜂、甲虫、毛毛虫、蚯蚓、地蜈蚣（耳蛾）、蚱蜢、飞蛾、蛞蝓、蜗牛、黄蜂、象鼻虫共12类害虫的图像，每张图像提供了YOLO格式的标注。

目录结构：
- **src/detection.py**：封装了YOLOv8检测模型的训练与加载。
- **src/dataset.py**：定义数据集类，将原始图像按照标注裁剪为单虫图片供分类模型训练。
- **src/classification.py**：封装ResNet50分类模型的训练、评估和推理功能。
- **main.py**：主脚本，串联以上模块，根据参数执行训练/推理流程。
- **requirements.txt**：依赖库列表（PyTorch、Torchvision、Ultralytics、OpenCV等）。
- **README.md**：使用说明和项目细节（假定放置了更详细的说明）。

数据准备：  
将AgroPest-12数据集下载并放置于 `data/AgroPest-12` 目录下。目录应包含以下结构：
```text
AgroPest-12/
├── images/
│   ├── train/  (训练集图像)
│   ├── val/    (验证集图像)
│   └── test/   (测试集图像, 如有)
└── labels/
    ├── train/  (训练集YOLO标注txt文件)
    ├── val/    (验证集标注)
    └── test/   (测试集标注, 如有)