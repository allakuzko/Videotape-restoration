from pathlib import Path
import shutil

SRC = Path("/home/alla/Завантажене/TAPE/train/gt/videos")
DST = Path("/home/alla/Завантажене/TAPE/all_videos")

DST.mkdir(exist_ok=True)

for video in SRC.glob("*.mp4"):
    shutil.copy(video, DST / video.name)

print("✅ Всі відео скопійовані")
