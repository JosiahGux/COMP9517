
import argparse, yaml, os
from PIL import Image
from src.utils.datasets import DetectionDataset

def save_crop(img_path, box, out_path, size=224):
    img = Image.open(img_path).convert('RGB')
    W,H = img.size
    x1,y1,x2,y2 = [int(max(0,min(v, W if i%2==0 else H))) for i,v in enumerate(box)]
    crop = img.crop((x1,y1,x2,y2)).resize((size,size))
    crop.save(out_path)

def main(cfg):
    os.makedirs("crops/train", exist_ok=True)
    ds = DetectionDataset(cfg["dataset"]["train_images"], cfg["dataset"]["train_labels"])
    # export GT crops
    for i in range(len(ds)):
        img, tgt, name = ds[i]
        img_path = os.path.join(cfg["dataset"]["train_images"], name)
        for j, (b, lab) in enumerate(zip(tgt["boxes"].numpy(), tgt["labels"].numpy())):
            out = os.path.join("crops/train", f"{lab-1}_{os.path.splitext(name)[0]}_{j}.jpg")
            save_crop(img_path, b, out, size=224)

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    main(cfg)
