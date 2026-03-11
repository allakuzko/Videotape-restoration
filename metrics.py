import cv2
import numpy as np
from pathlib import Path
import math

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * math.log10(255.0 / math.sqrt(mse))

gt_dir = Path("train/frames/gt")
restored_dir = Path("restored_frames")

files = sorted(restored_dir.glob("*.png"))

scores = []

for f in files:
    gt = cv2.imread(str(gt_dir / f.name))
    restored = cv2.imread(str(f))

    score = psnr(gt, restored)
    scores.append(score)

print("Average PSNR:", sum(scores) / len(scores))
