import numpy as np
import cv2


def psnr(img1, img2):
    try:
        if img1.ndim == img2.ndim:
            nd = img1.ndim
        else:
            raise ValueError("img1 and img2 must have the same number of dimensions")
        return cv2.PSNR(np.copy(img1[:, :, 0:nd]).astype(np.float32), np.copy(img2[:, :, 0:nd]).astype(np.float32))
    except:
        return cv2.PSNR(np.copy(img1).astype(np.float32), np.copy(img2).astype(np.float32))

def mse(img1, img2):
    height, width = img1.shape[:2]
    return np.sum((img1.astype(np.float64) - img2.astype(np.float64)) ** 2) / float(height * width)