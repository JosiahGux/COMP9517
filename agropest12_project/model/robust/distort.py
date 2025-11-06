
import numpy as np, cv2

def add_gaussian_noise(image, std=0.1):
    noise = np.random.normal(0, std, image.shape)
    x = image + noise
    return np.clip(x, 0.0, 1.0)

def add_blur(image, ksize=5):
    img = (image*255).astype(np.uint8)
    x = cv2.GaussianBlur(img, (ksize|1, ksize|1), 0)
    return x.astype(np.float32)/255.0

def adjust_brightness(image, factor=0.5):
    return np.clip(image*factor, 0.0, 1.0)

def add_occlusion(image, occ=0.2):
    H,W,_ = image.shape
    s = int(min(H,W)*occ)
    x0, y0 = np.random.randint(0, W-s), np.random.randint(0, H-s)
    x = image.copy(); x[y0:y0+s, x0:x0+s, :] = 0
    return x
