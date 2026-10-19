import numpy as np, cv2
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

IMG_SIZE = (64, 64)

def prepare_image(img_rgb):
    """Image RGB uint8 -> redimensionnée (64x64) et normalisée dans [0, 1]."""
    img = cv2.resize(img_rgb, IMG_SIZE)
    return img.astype('float32') / 255.0

def _u8(img):
    return np.clip(np.round(img * 255), 0, 255).astype('uint8')

def rgb_stats(img):
    return np.concatenate([img.mean(axis=(0, 1)), img.std(axis=(0, 1))])            # 6

def hsv_hist(img, bins=8):
    hsv = cv2.cvtColor(_u8(img), cv2.COLOR_RGB2HSV)
    h = [cv2.calcHist([hsv], [i], None, [bins], [0, 256 if i else 180]).flatten() for i in range(3)]
    h = np.concatenate(h)
    return h / (h.sum() + 1e-7)                                                     # 24

def glcm_feats(img):
    g = cv2.cvtColor(_u8(img), cv2.COLOR_RGB2GRAY)
    m = graycomatrix(g, [1], [0], levels=256, symmetric=True, normed=True)
    return np.array([graycoprops(m, p)[0, 0] for p in
                     ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']])   # 5

def lbp_hist(img, P=8, R=1):
    g = cv2.cvtColor(_u8(img), cv2.COLOR_RGB2GRAY)
    l = local_binary_pattern(g, P, R, method='uniform')
    h, _ = np.histogram(l, bins=P + 2, range=(0, P + 2))
    return h / (h.sum() + 1e-7)                                                     # 10

def hu_moments(img):
    g = cv2.cvtColor(_u8(img), cv2.COLOR_RGB2GRAY)
    hu = cv2.HuMoments(cv2.moments(g)).flatten()
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)                              # 7

def extract(img):
    return np.concatenate([rgb_stats(img), hsv_hist(img), glcm_feats(img), lbp_hist(img), hu_moments(img)])
