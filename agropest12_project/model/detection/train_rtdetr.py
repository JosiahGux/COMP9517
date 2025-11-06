
import argparse, yaml, os, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from agropest12_project.model.utils.seed import set_seed
from agropest12_project.model.utils.datasets import DetectionDataset
from agropest12_project.model.detection.rtdetr_model import ToyRTDETR

def to_float(x, name):
    try:
        return float(x)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be float，current value: {x!r} (type={type(x).__name__})")

"""
def main(cfg):
    set_seed(cfg["training"]["seed"])
    train = DetectionDataset(cfg["dataset"]["train_images"], cfg["dataset"]["train_labels"])
    dl = DataLoader(train, batch_size=cfg["training"]["det"]["batch_size"], shuffle=True, collate_fn=lambda x: x)
    model = ToyRTDETR(num_classes=cfg["dataset"]["classes"]+1, num_queries=cfg["model"]["num_queries"]).cuda()
    #opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["det"]["lr"], weight_decay=cfg["training"]["det"]["weight_decay"])
    opt = torch.optim.AdamW(model.parameters(), lr=to_float(cfg['training']['det']['lr'],"training.det.lr"), weight_decay=to_float(cfg['training']['det']['weight_decay'],"training.det.weight_decay"))

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
"""

def main(cfg):
    set_seed(cfg["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train = DetectionDataset(cfg["dataset"]["train_images"], cfg["dataset"]["train_labels"])
    dl = DataLoader(train, batch_size=cfg["training"]["det"]["batch_size"], shuffle=True, collate_fn=lambda x: x)

    model = ToyRTDETR(
        num_classes=cfg["dataset"]["classes"] + 1,
        num_queries=cfg["model"]["num_queries"]
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=to_float(cfg['training']['det']['lr'], "training.det.lr"),
        weight_decay=to_float(cfg['training']['det']['weight_decay'], "training.det.weight_decay")
    )

    os.makedirs(cfg["output"]["save_dir"], exist_ok=True)
    for epoch in range(cfg["training"]["det"]["epochs"]):
        pbar = tqdm(dl, desc=f"RT-DETR Epoch {epoch+1}")
        for batch in pbar:
            imgs, targets, _ = zip(*batch)
            imgs = [img.to(device) for img in imgs]

            opt.zero_grad()

            outs = model(imgs)   # 期望是 list[dict], 每个 dict 里有 "scores": Tensor

            # —— 关键修复（保证 loss 始终是 Tensor）——
            per_out_losses = []
            for out in outs:
                s = out.get("scores", None)
                if isinstance(s, torch.Tensor) and s.numel() > 0:
                    per_out_losses.append(1.0 - s.mean())              # Tensor
                else:
                    per_out_losses.append(torch.zeros((), device=device))  # 标量 Tensor（非 float）

            num_outs = max(len(outs), 1)
            loss = torch.stack(per_out_losses).sum() / num_outs           # 仍是 Tensor

            loss.backward()
            opt.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")  # 仅显示时再转 float

    torch.save(model.state_dict(), os.path.join(cfg["output"]["save_dir"], "rtdetr_toy.pth"))


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--lr", type=float)  # ✅ 指定 float
    ap.add_argument("--weight-decay", type=float)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    main(cfg)
