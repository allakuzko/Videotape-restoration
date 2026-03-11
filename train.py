import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.tape_dataset import TapedVideoFrameDataset
from models.unet import UNetLight
from torchmetrics.image import StructuralSimilarityIndexMeasure


def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ----- Dataset -----
    dataset = TapedVideoFrameDataset(
        input_dir="train/frames/input",
        gt_dir="train/frames/gt"
    )

    # беремо 1000 кадрів
    dataset.files = dataset.files[:1000]

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )

    # ----- Model -----
    model = UNetLight().to(device)

    # ----- Loss -----
    l1_loss = nn.L1Loss()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # ----- Optimizer -----
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 3

    for epoch in range(num_epochs):

        model.train()
        epoch_loss = 0.0

        loop = tqdm(loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")

        for x, y in loop:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            preds = model(x)

            # ----- комбінований loss -----
            l1 = l1_loss(preds, y)
            ssim = ssim_metric(preds, y)

            loss = l1 + 0.2 * (1 - ssim)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            loop.set_postfix(
                l1=l1.item(),
                ssim=ssim.item(),
                loss=loss.item()
            )

        avg_loss = epoch_loss / len(loader)
        print(f"\nEpoch {epoch+1}: Avg Loss = {avg_loss:.4f}\n")

    # ----- Save model -----
    torch.save(model.state_dict(), "unet_tape.pth")
    print("✅ Model saved as unet_tape.pth")


if __name__ == "__main__":
    train()
