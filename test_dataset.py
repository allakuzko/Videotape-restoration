from datasets.tape_dataset import TapedVideoFrameDataset

dataset = TapedVideoFrameDataset(
    input_dir="train/frames/input",
    gt_dir="train/frames/gt"
)

print("Dataset size:", len(dataset))

x, y = dataset[0]
print("Input shape:", x.shape)
print("GT shape:", y.shape)
print("Input min/max:", x.min().item(), x.max().item())
