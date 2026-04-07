import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import time
import json
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from pycocotools import mask as mask_utils

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")


class SA1BDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_ids = [f[:-4] for f in os.listdir(data_dir) if f.endswith('.jpg')]
        
    def __len__(self):
        return len(self.file_ids)
        
    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        img_path = os.path.join(self.data_dir, f"{file_id}.jpg")
        json_path = os.path.join(self.data_dir, f"{file_id}.json")
        
        image = Image.open(img_path).convert('RGB')
        
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        ann = data['annotations'][0]
        
        x, y, w, h = ann['bbox']
        input_box = np.array([x, y, x + w, y + h])
        
        rle = ann['segmentation']
        gt_mask = mask_utils.decode(rle).astype(bool)
        
        return image, input_box, gt_mask

def custom_collate_fn(batch):
    """
    Since batch_size=1 and images vary in size, we bypass default collation
    and just return lists of the original objects.
    """
    images = [item[0] for item in batch]
    boxes = [item[1] for item in batch]
    masks = [item[2] for item in batch]
    return images, boxes, masks

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

np.random.seed(3)

# Defining dataset and dataloader
dataset = SA1BDataset(data_dir="benchmark_dataset")
dataloader = DataLoader(
    dataset, 
    batch_size=1,
    shuffle=False, 
    num_workers=4, 
    pin_memory=False,
    collate_fn=custom_collate_fn
)

# Running warmup experiments
image = Image.open(f"{sam3_root}/assets/images/truck.jpg")
bpe_path = f"sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
model = build_sam3_image_model(bpe_path=bpe_path, enable_inst_interactivity=True)

processor = Sam3Processor(model)
input_box = np.array([425, 600, 700, 875])

print("Warming up GPU kernels...")
num_warmup = 10
with torch.no_grad():
    for _ in range(num_warmup):
        inference_state = processor.set_image(image)
        _ = model.predict_inst(
            inference_state,
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )

# Running experiments
print("Running benchmark experiments...")
encoder_latencies = []
decoder_latencies = []
iou_scores = []

# Reset VRAM tracking before the actual benchmark starts
if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()

with torch.no_grad():
    for images, input_boxes, gt_masks in tqdm(dataloader):
        image, input_box, gt_mask = images[0], input_boxes[0], gt_masks[0]
        if device.type == "cuda": 
            torch.cuda.synchronize()

        start_enc = time.perf_counter()
        
        inference_state = processor.set_image(image)
        
        if device.type == "cuda": 
            torch.cuda.synchronize()

        end_enc = time.perf_counter()
        encoder_latencies.append((end_enc - start_enc) * 1000)

        # Benchmark B: Mask Decoder (The lightweight prompt network)
        if device.type == "cuda": 
            torch.cuda.synchronize()

        start_dec = time.perf_counter()
        
        masks, scores, _ = model.predict_inst(
            inference_state,
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        
        if device.type == "cuda": 
            torch.cuda.synchronize()

        end_dec = time.perf_counter()
        decoder_latencies.append((end_dec - start_dec) * 1000)

        pred_mask = masks[0] 
        
        # Ensure masks are boolean (threshold at 0.0 if SAM returns logits instead of bools)
        if hasattr(pred_mask, 'dtype') and pred_mask.dtype != bool: # For numpy
            pred_mask = pred_mask > 0.0
            
        # Calculate intersection and union
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        
        # Compute IoU, avoiding division by zero
        iou = intersection / union if union > 0 else 0.0
        iou_scores.append(iou)


print("\n" + "="*30)
print(" BENCHMARK RESULTS")
print("="*30)
print(f"Total Experiments: {len(dataset)}")


if device.type == "cuda":
    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
    print("\n[Hardware Utilization]")
    print(f"Peak VRAM:    {peak_vram:.2f} GB")


print("\n[Image Encoder Performance]")
print(f"Mean Latency: {np.mean(encoder_latencies):.2f} ms")
print(f"p95 Latency:  {np.percentile(encoder_latencies, 95):.2f} ms")
print(f"p99 Latency:  {np.percentile(encoder_latencies, 99):.2f} ms")
print(f"Throughput:   {1000 / np.mean(encoder_latencies):.2f} images/sec")

print("\n[Mask Decoder Performance]")
print(f"Mean Latency: {np.mean(decoder_latencies):.2f} ms")
print(f"p95 Latency:  {np.percentile(decoder_latencies, 95):.2f} ms")
print(f"p99 Latency:  {np.percentile(decoder_latencies, 99):.2f} ms")
print(f"Throughput:   {1000 / np.mean(decoder_latencies):.2f} prompts/sec")

print("\n[IoU Performance]")
print(f"Mean IoU: {np.mean(iou_scores):.4f}")
print(f"p5 IoU (Worst 5%):  {np.percentile(iou_scores, 5):.4f}") # For accuracy, lower percentiles are usually more informative
print(f"p95 IoU:  {np.percentile(iou_scores, 95):.4f}")
print(f"p99 IoU:  {np.percentile(iou_scores, 99):.4f}")


print("\nGenerating and saving distribution plots...")

# Optional: Create a directory to keep things organized
os.makedirs("benchmark_results", exist_ok=True)

# 1. Encoder Latency Distribution
plt.figure(figsize=(10, 6))
plt.hist(encoder_latencies, bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Image Encoder Latency')
plt.xlabel('Latency (ms)')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.savefig('benchmark_results/encoder_latency_dist.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Decoder Latency Distribution
plt.figure(figsize=(10, 6))
plt.hist(decoder_latencies, bins=50, color='lightgreen', edgecolor='black')
plt.title('Distribution of Mask Decoder Latency')
plt.xlabel('Latency (ms)')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.savefig('benchmark_results/decoder_latency_dist.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. IoU Score Distribution
plt.figure(figsize=(10, 6))
# Using range=(0,1) ensures the x-axis represents the full possible IoU range
plt.hist(iou_scores, bins=50, range=(0, 1), color='salmon', edgecolor='black') 
plt.title('Distribution of IoU Scores')
plt.xlabel('Intersection over Union (IoU)')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.savefig('benchmark_results/iou_score_dist.png', dpi=300, bbox_inches='tight')
plt.close()

print("Plots successfully saved to 'benchmark_results/' directory.")