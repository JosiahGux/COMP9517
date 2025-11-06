
import torch, torch.nn as nn
import torchvision
class ToyRTDETR(nn.Module):
    """Toy RT-DETR-like detector: ResNet50 backbone + Transformer encoder/decoder + heads.
    NOTE: For coursework demonstration; not a faithful reproduction.
    """
    def __init__(self, num_classes=13, num_queries=100, d_model=256):
        super().__init__()
        backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # C5
        self.input_proj = nn.Conv2d(2048, d_model, 1)
        self.pos = nn.Parameter(torch.randn(1, d_model, 32, 32))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=1024, batch_first=True)
        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=8, dim_feedforward=1024, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, 4)
        self.decoder = nn.TransformerDecoder(dec_layer, 4)
        self.query = nn.Parameter(torch.randn(num_queries, d_model))
        self.cls  = nn.Linear(d_model, num_classes)
        self.box  = nn.Sequential(nn.Linear(d_model, 4), nn.Sigmoid())
    def forward(self, images):
        if isinstance(images, (list,tuple)):
            images = torch.stack(images,0)
        feats = self.backbone(images)              # [B,2048,H,W]
        src = self.input_proj(feats)               # [B,d,H,W]
        B,C,H,W = src.shape
        pos = nn.functional.interpolate(self.pos, size=(H,W), mode="bilinear", align_corners=False)
        src = (src+pos).flatten(2).transpose(1,2)  # [B,HW,d]
        mem = self.encoder(src)                    # [B,HW,d]
        q = self.query.unsqueeze(0).expand(B,-1,-1)# [B,Q,d]
        hs = self.decoder(q, mem)                  # [B,Q,d]
        logits = self.cls(hs); boxes = self.box(hs)
        out = []
        for b in range(B):
            probs = logits[b].softmax(-1); scores, labels = probs.max(-1)
            mask = (labels!=0) & (scores>0.5)
            bb = boxes[b][mask]; sc = scores[mask]; lb = labels[mask]
            # convert to xyxy assuming boxes are normalized cxcywh
            if bb.numel()>0:
                cx, cy, w, h = bb.unbind(-1)
                x1 = (cx - w/2); y1 = (cy - h/2); x2 = (cx + w/2); y2 = (cy + h/2)
                bb = torch.stack([x1,y1,x2,y2], -1)
            out.append({"boxes": bb, "scores": sc, "labels": lb})
        return out
