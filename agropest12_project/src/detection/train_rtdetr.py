
import argparse, yaml, os, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.utils.seed import set_seed
from src.utils.datasets import DetectionDataset
from src.detection.rtdetr_model import ToyRTDETR

def main(cfg):
    set_seed(cfg["training"]["seed"])
    train = DetectionDataset(cfg["dataset"]["train_images"], cfg["dataset"]["train_labels"])
    dl = DataLoader(train, batch_size=cfg["training"]["det"]["batch_size"], shuffle=True, collate_fn=lambda x: x)
    model = ToyRTDETR(num_classes=cfg["dataset"]["classes"]+1, num_queries=cfg["model"]["num_queries"]).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["det"]["lr"], weight_decay=cfg["training"]["det"]["weight_decay"])
    os.makedirs(cfg["output"]["save_dir"], exist_ok=True)
    for epoch in range(cfg["training"]["det"]["epochs"]):
        pbar = tqdm(dl, desc=f"RT-DETR Epoch {epoch+1}")
        for batch in pbar:
            imgs, targets, _ = zip(*batch)
            imgs = [img.cuda() for img in imgs]
            # Dummy loss: encourage more predictions with higher scores (placeholder)
            outs = model(imgs)
            loss = sum([(1-out["scores"].mean()) for out in outs if out["scores"].numel()>0]) / max(len(outs),1)
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix(loss=float(loss))
    torch.save(model.state_dict(), os.path.join(cfg["output"]["save_dir"], "rtdetr_toy.pth"))

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    main(cfg)
