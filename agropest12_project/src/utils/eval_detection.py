
import argparse, yaml, os, torch
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights
from torch.utils.data import DataLoader
from src.utils.datasets import DetectionDataset
from src.utils.metrics import iou, voc_map
from tqdm import tqdm

def eval_retinanet(cfg):
    ds = DetectionDataset(cfg["dataset"]["val_images"], cfg["dataset"]["val_labels"])
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=lambda x: x)
    num_classes = cfg["dataset"]["classes"]
    model = retinanet_resnet50_fpn(weights=None, num_classes=num_classes+1).cuda().eval()
    ckpt = os.path.join(cfg["output"]["save_dir"], "retinanet_res50_agropest12.pth")
    model.load_state_dict(torch.load(ckpt, map_location="cuda"))
    preds_by_cls = {c: [] for c in range(1, num_classes+1)}
    gt_counts = {c: 0 for c in range(1, num_classes+1)}
    for batch in tqdm(dl):
        img, target, _ = batch[0]
        img = img.cuda()
        out = model([img])[0]
        gt_boxes = target["boxes"].numpy(); gt_labels = target["labels"].numpy()
        for c in range(1, num_classes+1):
            gt_counts[c] += (gt_labels==c).sum()
        for box, score, lab in zip(out["boxes"].cpu().numpy(), out["scores"].cpu().numpy(), out["labels"].cpu().numpy()):
            if lab==0 or score<0.5: continue
            # match to best GT of same class
            iou_max, match = 0, -1
            for j, g in enumerate(gt_boxes):
                if gt_labels[j]==lab:
                    ii = iou(box, g)
                    if ii > iou_max:
                        iou_max, match = ii, j
            preds_by_cls[lab].append((float(score), iou_max>=0.5))
    mAP, APs = voc_map(preds_by_cls, gt_counts)
    print("mAP@0.5:", round(mAP,4), "APs:", {k:round(v,3) for k,v in APs.items()})

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    eval_retinanet(cfg)
