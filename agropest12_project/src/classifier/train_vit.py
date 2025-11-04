
import argparse, yaml, os, torch, numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from src.utils.seed import set_seed
from src.utils.datasets import ClassificationDataset
from src.classifier.vit_model import ViTClassifier

def main(cfg):
    set_seed(cfg["training"]["seed"])
    ds = ClassificationDataset("crops/train")
    labels = np.array([y for _, y in [ds[i] for i in range(len(ds))]])
    counts = np.bincount(labels, minlength=len(ds.id2lbl))
    weights = (1.0 / np.maximum(counts, 1))[labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    dl = DataLoader(ds, batch_size=cfg['training']['cls']['batch_size'], sampler=sampler)
    model = ViTClassifier(num_classes=len(ds.id2lbl)).cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['training']['cls']['lr'], weight_decay=cfg['training']['cls']['weight_decay'])
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
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    main(cfg)
