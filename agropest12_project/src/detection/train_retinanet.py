
import argparse, yaml, os, torch
from torch.utils.data import DataLoader
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights
from tqdm import tqdm
from src.utils.seed import set_seed
from src.utils.datasets import DetectionDataset

def main(cfg):
    set_seed(cfg["training"]["seed"])
    train = DetectionDataset(cfg["dataset"]["train_images"], cfg["dataset"]["train_labels"])
    dl = DataLoader(train, batch_size=cfg["training"]["det"]["batch_size"], shuffle=True, collate_fn=lambda x: x)
    num_classes = cfg["dataset"]["classes"]+1
    model = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT, num_classes=num_classes).cuda().train()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    os.makedirs(cfg["output"]["save_dir"], exist_ok=True)
    for epoch in range(cfg["training"]["det"]["epochs"]):
        pbar = tqdm(dl, desc=f"RetinaNet Epoch {epoch+1}")
        for batch in pbar:
            imgs, targets, _ = zip(*batch)
            imgs = [img.cuda() for img in imgs]
            targets = [{k:v.cuda() for k,v in t.items()} for t in targets]
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix({k: float(v) for k,v in loss_dict.items()})
    torch.save(model.state_dict(), os.path.join(cfg["output"]["save_dir"], "retinanet_res50_agropest12.pth"))

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    main(cfg)
