from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

INPUT_DIR = Path("train/frames/input")
GT_DIR = Path("train/frames/gt")
OUT_DIR = Path("eda_results")

OUT_DIR.mkdir(exist_ok=True)

# ----------------------------
# 1. Вибір випадкових кадрів
# ----------------------------
files = sorted(INPUT_DIR.glob("*.png"))
samples = random.sample(files, 5)

# ----------------------------
# 2. Візуальне порівняння
# ----------------------------
for f in samples:
    name = f.name
    inp = cv2.cvtColor(cv2.imread(str(INPUT_DIR / name)), cv2.COLOR_BGR2RGB)
    gt = cv2.cvtColor(cv2.imread(str(GT_DIR / name)), cv2.COLOR_BGR2RGB)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(inp)
    axs[0].set_title("Input (degraded)")
    axs[0].axis("off")

    axs[1].imshow(gt)
    axs[1].set_title("GT (restored)")
    axs[1].axis("off")

    plt.tight_layout()
    plt.savefig(OUT_DIR / f"compare_{name}")
    plt.close()

# ----------------------------
# 3. Гістограми кольорів
# ----------------------------
def plot_hist(image, title, path):
    colors = ("r", "g", "b")
    plt.figure()
    for i, c in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=c)
    plt.title(title)
    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")
    plt.savefig(path)
    plt.close()

for f in samples[:3]:
    name = f.name
    inp = cv2.imread(str(INPUT_DIR / name))
    gt = cv2.imread(str(GT_DIR / name))

    plot_hist(inp, f"Input histogram {name}", OUT_DIR / f"hist_input_{name}.png")
    plot_hist(gt, f"GT histogram {name}", OUT_DIR / f"hist_gt_{name}.png")

# ----------------------------
# 4. Яскравість і контраст
# ----------------------------
def brightness_contrast(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.mean(), gray.std()

stats = []

for f in samples:
    name = f.name
    inp = cv2.imread(str(INPUT_DIR / name))
    gt = cv2.imread(str(GT_DIR / name))

    b_i, c_i = brightness_contrast(inp)
    b_g, c_g = brightness_contrast(gt)

    stats.append((name, b_i, c_i, b_g, c_g))

# Збережемо статистику
with open(OUT_DIR / "brightness_contrast.txt", "w") as f:
    for s in stats:
        f.write(
            f"{s[0]} | "
            f"Input: brightness={s[1]:.2f}, contrast={s[2]:.2f} | "
            f"GT: brightness={s[3]:.2f}, contrast={s[4]:.2f}\n"
        )

print("✅ EDA завершено. Результати в папці eda_results/")
