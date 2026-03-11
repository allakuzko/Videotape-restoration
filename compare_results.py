import cv2
import numpy as np
from pathlib import Path
import math
import matplotlib.pyplot as plt

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * math.log10(255.0 / math.sqrt(mse))

input_dir = Path("train/frames/input")
gt_dir = Path("train/frames/gt")
restored_dir = Path("restored_frames")

out_dir = Path("comparison_results")
out_dir.mkdir(exist_ok=True)

files = sorted(restored_dir.glob("*.png"))[:10]

improvements = []

for f in files:
    name = f.name

    inp = cv2.imread(str(input_dir / name))
    gt = cv2.imread(str(gt_dir / name))
    restored = cv2.imread(str(restored_dir / name))

    psnr_input = psnr(inp, gt)
    psnr_restored = psnr(restored, gt)

    improvement = psnr_restored - psnr_input
    improvements.append(improvement)

    # Створюємо горизонтальне порівняння
    combined = np.hstack([inp, restored, gt])

    cv2.putText(combined, f"Input PSNR: {psnr_input:.2f}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, (0,255,0), 2)

    cv2.putText(combined, f"Restored PSNR: {psnr_restored:.2f}", 
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, (0,255,0), 2)

    cv2.imwrite(str(out_dir / name), combined)

print("Average PSNR improvement:", sum(improvements)/len(improvements))
print("Comparison images saved in comparison_results/")
