
import numpy as np, torch

def iou(box1, box2):
    xA = max(box1[0], box2[0]); yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2]); yB = min(box1[3], box2[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    inter = interW * interH
    if inter <= 0: return 0.0
    a1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    a2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    return inter / (a1 + a2 - inter + 1e-9)

def voc_map(preds_by_cls, gt_counts):
    APs = {}
    for cls, preds in preds_by_cls.items():
        if gt_counts.get(cls,0) == 0 or len(preds)==0:
            continue
        preds.sort(key=lambda x: x[0], reverse=True)
        tp=0; fp=0; prec=[]; rec=[]; total=gt_counts[cls]
        for score, is_tp in preds:
            if is_tp: tp += 1
            else: fp += 1
            prec.append(tp / max(tp+fp,1e-9))
            rec.append(tp / max(total,1e-9))
        # 11-point interpolation (simple)
        ap = 0.0
        for t in np.linspace(0,1,11):
            p = max([p for p,r in zip(prec,rec) if r>=t] + [0])
            ap += p/11
        APs[cls] = ap
    mAP = np.mean(list(APs.values())) if APs else 0.0
    return mAP, APs
