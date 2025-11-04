
import argparse, yaml, torch, os
import torchvision.transforms as T
from PIL import Image
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights

def main(cfg, image):
    num_classes = cfg["dataset"]["classes"]+1
    model = retinanet_resnet50_fpn(weights=None, num_classes=num_classes).cuda().eval()
    model.load_state_dict(torch.load(os.path.join(cfg["output"]["save_dir"], "retinanet_res50_agropest12.pth"), map_location="cuda"))
    img = Image.open(image).convert('RGB')
    tens = T.ToTensor()(img).cuda()
    with torch.no_grad():
        out = model([tens])[0]
    for b,s,l in zip(out["boxes"].cpu().numpy(), out["scores"].cpu().numpy(), out["labels"].cpu().numpy()):
        if s<0.5 or l==0: continue
        print(f"Box {b.tolist()} score {float(s):.2f} class {int(l)-1}")

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--image", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    main(cfg, args.image)
