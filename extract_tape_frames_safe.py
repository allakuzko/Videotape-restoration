import cv2
from pathlib import Path
from tqdm import tqdm

ROOT = Path("/home/alla/Завантажене/TAPE/train")

IN_VIDEOS = ROOT / "input/videos"
GT_VIDEOS = ROOT / "gt/videos"

OUT_ROOT = ROOT / "frames"
OUT_IN = OUT_ROOT / "input"
OUT_GT = OUT_ROOT / "gt"

OUT_IN.mkdir(parents=True, exist_ok=True)
OUT_GT.mkdir(parents=True, exist_ok=True)

def extract(video_dir, out_dir):
    for video in tqdm(sorted(video_dir.glob("*.mp4")), desc=video_dir.name):
        cap = cv2.VideoCapture(str(video))
        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            name = f"{video.stem}_{frame_id:06d}.png"
            cv2.imwrite(str(out_dir / name), frame)
            frame_id += 1

        cap.release()

extract(IN_VIDEOS, OUT_IN)
extract(GT_VIDEOS, OUT_GT)

print("✅ Кадри витягнуті з відео")
