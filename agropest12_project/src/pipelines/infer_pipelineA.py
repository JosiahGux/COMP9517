
import argparse, yaml, torch, os
import torchvision.transforms as T
from PIL import Image
from src.detection.rtdetr_model import ToyRTDETR
from src.classifier.vit_model import ViTClassifier

def crop_and_prepare(img, box, size=224):
    W,H = img.size
    x1,y1,x2,y2 = [int(max(0,min(v, W if i%2==0 else H))) for i,v in enumerate(box)]
    crop = img.crop((x1,y1,x2,y2)).resize((size,size))
    transform = T.Compose([T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    return transform(crop)

def main(cfg, image):
    det = ToyRTDETR(num_classes=cfg["dataset"]["classes"]+1, num_queries=cfg["model"]["num_queries"]).cuda().eval()
    det.load_state_dict(torch.load(os.path.join(cfg["output"]["save_dir"], "rtdetr_toy.pth"), map_location="cuda"))
    cls = ViTClassifier(num_classes=cfg["dataset"]["classes"]).cuda().eval()
    cls.load_state_dict(torch.load(os.path.join(cfg["output"]["save_dir"], "vit_b16_classifier_agropest12.pth"), map_location="cuda"))
    img = Image.open(image).convert('RGB')
    tens = T.ToTensor()(img).unsqueeze(0).cuda()
    outs = det(tens)[0]
    for b, s, l in zip(outs["boxes"], outs["scores"], outs["labels"]):
        if s<0.5: continue
        crop = crop_and_prepare(img, (b*torch.tensor([img.size[0],img.size[1],img.size[0],img.size[1]]).cuda()).tolist(), 224)
        with torch.no_grad():
            pred = cls(crop.unsqueeze(0).cuda()).softmax(-1).argmax(-1).item()
        print(f"Box {b.tolist()} score {float(s):.2f} -> class {pred}")

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--image", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    main(cfg, args.image)
