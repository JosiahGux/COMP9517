
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torch, os

class DetectionDataset(Dataset):
    """YOLO format labels (per-image .txt): class cx cy w h (normalized)."""
    def __init__(self, img_dir, lbl_dir, transform=None):
        self.img_dir, self.lbl_dir = img_dir, lbl_dir
        self.imgs = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))])
        self.transform = transform or (lambda x: T.ToTensor()(x))
    def __len__(self): return len(self.imgs)
    def __getitem__(self, idx):
        name = self.imgs[idx]
        img = Image.open(os.path.join(self.img_dir, name)).convert('RGB')
        W, H = img.size
        # load labels (allow missing -> no objects)
        label_path = os.path.splitext(os.path.join(self.lbl_dir, name))[0] + ".txt"
        boxes, labels = [], []
        if os.path.exists(label_path):
            for line in open(label_path):
                vals = line.strip().split()
                if len(vals) != 5: continue
                cls, cx, cy, w, h = map(float, vals)
                labels.append(int(cls)+1)  # shift by 1; 0 reserved for background
                x1 = (cx - w/2) * W; y1 = (cy - h/2) * H
                x2 = (cx + w/2) * W; y2 = (cy + h/2) * H
                boxes.append([x1,y1,x2,y2])
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }
        return self.transform(img), target, name

class ClassificationDataset(Dataset):
    """Crops on disk, file name like <class>_<id>.jpg or subfolders per class."""
    def __init__(self, crop_dir, transform=None):
        self.items = []
        self.transform = transform or T.Compose([
            T.Resize((224,224)), T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        # accept flat folder with prefix or class subfolders
        for root, _, files in os.walk(crop_dir):
            cls_name = os.path.basename(root)
            for f in files:
                if not f.lower().endswith(('.jpg','.jpeg','.png')): continue
                if "_" in f:
                    label = f.split("_")[0]
                else:
                    label = cls_name
                self.items.append((os.path.join(root,f), label))
        # map labels to ids
        labels = sorted({lab for _,lab in self.items})
        self.lbl2id = {lab:i for i,lab in enumerate(labels)}
        self.id2lbl = labels
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        p, lab = self.items[idx]
        img = Image.open(p).convert('RGB')
        return self.transform(img), self.lbl2id[lab]
