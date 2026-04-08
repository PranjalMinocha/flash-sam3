import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

import huggingface_hub
import sam3
import sam3.perflib.fused as sam3_fused
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

config = {
    "bpe_path": "assets/bpe_simple_vocab_16e6.txt.gz",
    "data_dir": "assets",
    "lr": 1e-4,
    "epochs": 10,
    "batch_size": 1,
    "save_every": 500,
    "output_dir": "checkpoints",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 4,
}


class SA1BDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_ids = [f[:-4] for f in os.listdir(data_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        fid = self.file_ids[idx]
        image = Image.open(os.path.join(self.data_dir, f"{fid}.jpg")).convert("RGB")
        with open(os.path.join(self.data_dir, f"{fid}.json")) as f:
            ann = json.load(f)["annotations"][0]
        x, y, w, h = ann["bbox"]
        box = np.array([x, y, x + w, y + h], dtype=np.float32)
        return image, box


def collate_fn(batch):
    return [b[0] for b in batch], [b[1] for b in batch]


def build_student(bpe_path):
    student = build_sam3_image_model(bpe_path=bpe_path, enable_inst_interactivity=True)
    # TODO: replace student.backbone with TinyViT
    return student


def train():
    huggingface_hub.login(token=os.environ["HF_TOKEN"])

    device = torch.device(config["device"])
    print(f"device: {device}")

    if device.type == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    teacher = build_sam3_image_model(
        bpe_path=config["bpe_path"], enable_inst_interactivity=True
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = build_student(config["bpe_path"])
    student.train()

    proc = Sam3Processor(teacher)
    transform = proc.transform

    optimizer = torch.optim.AdamW(
        student.backbone.parameters(), lr=config["lr"]
    )

    dataset = SA1BDataset(config["data_dir"])
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        collate_fn=collate_fn,
    )

    os.makedirs(config["output_dir"], exist_ok=True)

    def preprocess(pil_image):
        img = TF.pil_to_tensor(pil_image).to(device)
        return transform(img).unsqueeze(0)

    step = 0
    for epoch in range(config["epochs"]):
        for images, _ in loader:
            img_tensor = preprocess(images[0])

            with torch.no_grad():
                t_embed = teacher.backbone.forward_image(img_tensor)["vision_features"].detach()

            s_embed = student.backbone.forward_image(img_tensor)["vision_features"]

            loss = F.mse_loss(s_embed, t_embed)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step % 50 == 0:
                print(f"epoch {epoch+1}  step {step}  loss {loss.item():.6f}")

            if step % config["save_every"] == 0:
                ckpt = os.path.join(config["output_dir"], f"student_step{step}.pt")
                torch.save(student.backbone.state_dict(), ckpt)
                print(f"saved {ckpt}")

    final = os.path.join(config["output_dir"], "student_final.pt")
    torch.save(student.backbone.state_dict(), final)
    print(f"training done — saved {final}")


if __name__ == "__main__":
    train()
