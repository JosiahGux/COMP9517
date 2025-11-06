
import argparse, yaml, os, torch, numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from agropest12_project.model.utils.seed import set_seed
from agropest12_project.model.utils.datasets import ClassificationDataset
from agropest12_project.model.classifier.vit_model import ViTClassifier
from pathlib import Path

def to_float(x, name):
    try:
        return float(x)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be float，current value: {x!r} (type={type(x).__name__})")

def main(cfg):
    set_seed(cfg["training"]["seed"])
    ds = ClassificationDataset("crops/train")
    labels = np.array([y for _, y in [ds[i] for i in range(len(ds))]])
    counts = np.bincount(labels, minlength=len(ds.id2lbl))
    weights = (1.0 / np.maximum(counts, 1))[labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    dl = DataLoader(ds, batch_size=cfg['training']['cls']['batch_size'], sampler=sampler)
    model = ViTClassifier(num_classes=len(ds.id2lbl)).cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=to_float(cfg['training']['cls']['lr'],"training.cls.lr"), weight_decay=to_float(cfg['training']['cls']['weight_decay'],"training.cls.weight_decay"))
    crit = torch.nn.CrossEntropyLoss()
    os.makedirs(cfg["output"]["save_dir"], exist_ok=True)
    for epoch in range(cfg['training']['cls']['epochs']):
        pbar = tqdm(dl, desc=f"ViT Epoch {epoch+1}")
        for x, y in pbar:
            x, y = x.cuda(), torch.tensor(y).cuda()
            logits = model(x)
            loss = crit(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix(loss=float(loss))
    torch.save(model.state_dict(), os.path.join(cfg["output"]["save_dir"], "vit_b16_classifier_agropest12.pth"))

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    #ap.add_argument("config", type=Path, metavar="CONFIG",help="训练配置文件路径（YAML/JSON）")
    ap.add_argument("--config", required=True)
    ap.add_argument("--lr", type=float)  # ✅ 指定 float
    ap.add_argument("--weight-decay", type=float)
    args = ap.parse_args()


    cfg = yaml.safe_load(open(args.config))
    main(cfg)
