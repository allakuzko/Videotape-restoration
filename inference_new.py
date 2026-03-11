from pathlib import Path
import cv2
import torch
import numpy as np
import math
from models.unet import UNetBig

# -----------------------
# PSNR
# -----------------------
def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * math.log10(255.0 / math.sqrt(mse))

# -----------------------
# Device
# -----------------------
device = torch.device("cpu")
print("Using device:", device)

# -----------------------
# Load model
# -----------------------
model = UNetBig().to(device)
model.load_state_dict(torch.load("unet_best_gpu.pth", map_location="cpu"))
model.eval()

# -----------------------
# Paths
# -----------------------
input_dir = Path("test/frames/input")
gt_dir = Path("test/frames/gt")

output_dir = Path("test_results")
output_dir.mkdir(exist_ok=True)

psnr_inputs = []
psnr_restored = []

# -----------------------
# Inference
# -----------------------
files = list(input_dir.rglob("*.png"))
files = files[:1000]
print("Testing on", len(files), "images")

for f in files:
    relative_path = f.relative_to(input_dir)

    inp = cv2.imread(str(f))
    gt = cv2.imread(str(gt_dir / relative_path))

    inp_resized = cv2.resize(inp, (256, 256))
    gt_resized = cv2.resize(gt, (256, 256))

    img = cv2.cvtColor(inp_resized, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0

    tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)

    out = out.squeeze(0).permute(1,2,0).cpu().numpy()
    out = (out * 255).clip(0,255).astype("uint8")
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    psnr_in = psnr(inp_resized, gt_resized)
    psnr_out = psnr(out_bgr, gt_resized)

    psnr_inputs.append(psnr_in)
    psnr_restored.append(psnr_out)

    combined = np.hstack([inp_resized, out_bgr, gt_resized])

    save_path = output_dir / relative_path
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(save_path), combined)


# -----------------------
# Results
# -----------------------
avg_input = sum(psnr_inputs) / len(psnr_inputs)
avg_restored = sum(psnr_restored) / len(psnr_restored)
improvement = avg_restored - avg_input

print("\n===== TEST RESULTS =====")
print("Average Input PSNR:", avg_input)
print("Average Restored PSNR:", avg_restored)
print("Average Improvement:", improvement)
