
import torch, numpy as np, cv2, matplotlib.pyplot as plt

def gradcam_for_class(model, image_tensor, target_class):
    model.eval()
    target = model.backbone.body.layer4
    feats = None
    grads = None
    def fwd(m, i, o): 
        nonlocal feats; feats = o
    def bwd(m, gi, go):
        nonlocal grads; grads = go[0]
    h1 = target.register_forward_hook(fwd)
    h2 = target.register_backward_hook(bwd)
    out = model([image_tensor.cuda()])[0]
    if len(out["scores"])==0:
        h1.remove(); h2.remove(); return None
    scores = out["scores"]; labels = out["labels"]
    if (labels==target_class).any():
        s = scores[labels==target_class].sum()
    else:
        s = scores[0]
    model.zero_grad(); s.backward()
    w = grads.mean(dim=(2,3), keepdim=True)
    cam = (w*feats).sum(1, keepdim=True).relu()[0,0].detach().cpu().numpy()
    cam = (cam - cam.min())/(cam.max()-cam.min()+1e-9)
    cam = cv2.resize(cam, (image_tensor.shape[3], image_tensor.shape[2]))
    h1.remove(); h2.remove()
    return cam
