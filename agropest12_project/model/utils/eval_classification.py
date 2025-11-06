
import argparse, yaml, torch, numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import torchvision.transforms as T
from agropest12_project.model.utils.datasets import ClassificationDataset
from agropest12_project.model.classifier.vit_model import ViTClassifier

def evaluate(cfg):
    ds = ClassificationDataset("crops/val")
    dl = DataLoader(ds, batch_size=64, shuffle=False)
    model = ViTClassifier(num_classes=len(ds.id2lbl)).cuda().eval()
    model.load_state_dict(torch.load("checkpoints/vit_b16_classifier_agropest12.pth", map_location="cuda"))
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in dl:
            x = x.cuda()
            all_logits.append(model(x).cpu())
            all_labels.append(y)
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    preds = logits.argmax(1)
    acc = (preds==labels).mean()
    # macro precision/recall/f1
    prec, rec, f1 = [], [], []
    K = len(ds.id2lbl)
    for c in range(K):
        tp = ((preds==c)&(labels==c)).sum()
        fp = ((preds==c)&(labels!=c)).sum()
        fn = ((preds!=c)&(labels==c)).sum()
        p = tp / (tp+fp+1e-9); r = tp/(tp+fn+1e-9)
        prec.append(p); rec.append(r); f1.append(2*p*r/(p+r+1e-9))
    # AUC macro (one-vs-rest)
    onehot = np.eye(K)[labels]
    probs = (logits - logits.max(1,keepdims=True))  # for numeric stability
    probs = np.exp(probs) / probs.sum(1, keepdims=True)
    auc = roc_auc_score(onehot, probs, average="macro", multi_class="ovr")
    print(f"Acc={acc:.4f}  P={np.mean(prec):.4f}  R={np.mean(rec):.4f}  F1={np.mean(f1):.4f}  AUC={auc:.4f}")

if __name__=="__main__":
    import sys, yaml
    cfg = yaml.safe_load(open(sys.argv[1])) if len(sys.argv)>1 else {}
    evaluate(cfg)
