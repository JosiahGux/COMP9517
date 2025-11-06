
from PIL import ImageDraw

def draw_boxes(pil_img, boxes, labels=None, color="red"):
    draw = ImageDraw.Draw(pil_img)
    for i, b in enumerate(boxes):
        x1,y1,x2,y2 = map(float, b)
        draw.rectangle([x1,y1,x2,y2], outline=color, width=2)
        if labels is not None:
            draw.text((x1, y1-10), str(labels[i]), fill=color)
    return pil_img
