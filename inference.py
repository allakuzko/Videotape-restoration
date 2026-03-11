import torch
import cv2
import numpy as np
import math
from pathlib import Path
from collections import defaultdict
from models.unet import UNetBig



def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * math.log10(255.0 / math.sqrt(mse))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----- Load model -----
model = UNetBig().to(device)
model.load_state_dict(torch.load("unet_best_gpu.pth", map_location="cpu"))
model.eval()


input_dir = Path("train/frames/input")
gt_dir = Path("train/frames/gt")
output_dir = Path("restored_frames")
compare_dir = Path("comparison_results")

output_dir.mkdir(exist_ok=True)
compare_dir.mkdir(exist_ok=True)

# очистка старих результатів
for f in output_dir.glob("*.png"):
    f.unlink()

for f in compare_dir.glob("*.png"):
    f.unlink()

# ----- Вибір кадрів з різних відео -----
files = sorted(input_dir.glob("*.png"))

selected = []
counter = defaultdict(int)

for f in files:
    vid = f.name.split("_")[0]

    if counter[vid] < 2:  # по 2 кадри з кожного відео
        selected.append(f)
        counter[vid] += 1

    if len(selected) >= 20:
        break

# ----- Inference -----
improvements = []
psnr_inputs = []
psnr_restored_list = []

for f in selected:
    name = f.name

    inp = cv2.imread(str(input_dir / name))
    gt = cv2.imread(str(gt_dir / name))

    # нормалізація
    img = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0

    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)

    out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out = (out * 255).clip(0, 255).astype("uint8")
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    # зберігаємо відновлений кадр
    cv2.imwrite(str(output_dir / name), out_bgr)

    # ----- Метрики -----
    psnr_input = psnr(inp, gt)
    psnr_restored = psnr(out_bgr, gt)

    improvement = psnr_restored - psnr_input

    psnr_inputs.append(psnr_input)
    psnr_restored_list.append(psnr_restored)
    improvements.append(improvement)

    # ----- Порівняльне зображення -----
    combined = np.hstack([inp, out_bgr, gt])

    cv2.putText(combined, f"Input PSNR: {psnr_input:.2f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)

    cv2.putText(combined, f"Restored PSNR: {psnr_restored:.2f}",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)

    cv2.putText(combined, f"Improvement: {improvement:.2f} dB",
                (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)

    cv2.imwrite(str(compare_dir / name), combined)

# ----- Підсумок -----
print("\n===== RESULTS =====")
print("Average Input PSNR:", sum(psnr_inputs) / len(psnr_inputs))
print("Average Restored PSNR:", sum(psnr_restored_list) / len(psnr_restored_list))
print("Average PSNR Improvement:", sum(improvements) / len(improvements))
print("\nComparison images saved in comparison_results/")
print("Restored frames saved in restored_frames/")
