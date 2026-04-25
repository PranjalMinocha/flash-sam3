import os
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import OneCycleLR

from torchvision import transforms
import torchvision.transforms.functional as TF

from transformers import Sam3Model, AutoProcessor
from efficientvit.sam_model_zoo import create_efficientvit_sam_model

class ResizeAndPad:
    """Resizes longest edge to target_size and statically pads the rest with zeros."""
    def __init__(self, target_size=1008):
        self.target_size = target_size

    def __call__(self, image):
        w, h = image.size
        scale = self.target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

        pad_w = self.target_size - new_w
        pad_h = self.target_size - new_h
        image = TF.pad(image, (0, 0, pad_w, pad_h), fill=0)
        
        return image

class SimpleImageDataset(Dataset):
    def __init__(self, data_dir, image_size=1008, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.data_dir = data_dir
        self.image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        self.transform = transforms.Compose([
            ResizeAndPad(target_size=image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std
            )
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        return self.transform(Image.open(img_path).convert('RGB'))


class OutputDistillationWrapper(nn.Module):
    def __init__(self, student_model, student_out_channels=256, teacher_out_channels=256):
        super().__init__()
        self.student = student_model
        
        if student_out_channels != teacher_out_channels:
            self.channel_projector = nn.Conv2d(student_out_channels, teacher_out_channels, kernel_size=1)
        else:
            self.channel_projector = nn.Identity()

    def forward(self, x, target_spatial_size):
        s_feats = self.student(x)
        s_feats = self.channel_projector(s_feats)
        
        if s_feats.shape[2:] != target_spatial_size:
            s_feats = F.interpolate(
                s_feats, 
                size=target_spatial_size, 
                mode='bilinear', 
                align_corners=False
            )
        return s_feats


def main():

    # Config
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPOCHS = 2
    ACCUMULATION_STEPS = 8
    LEARNING_RATE = 1e-4
    IMAGE_SIZE = 1008
    BATCH_SIZE = 2
    DATA_DIR = "sa_1b_dataset_0"
    MODEL_DIR = "facebook/sam3"
    VAL_SPLIT = 0.1

    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    image_mean = processor.image_processor.image_mean
    image_std = processor.image_processor.image_std   

    print("Initializing and Splitting Dataset...")
    full_dataset = SimpleImageDataset(
        data_dir=DATA_DIR, 
        image_size=IMAGE_SIZE, 
        mean=image_mean, 
        std=image_std
    )
    
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Total Images: {len(full_dataset)} | Training: {train_size} | Validation: {val_size}")

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print("Loading models...")
    sam3_model = Sam3Model.from_pretrained(MODEL_DIR, torch_dtype=torch.float16)
    teacher_encoder = sam3_model.vision_encoder.to(DEVICE).eval()
    
    for param in teacher_encoder.parameters():
        param.requires_grad = False

    student_encoder = create_efficientvit_sam_model(
        name="efficientvit-sam-l0", 
        pretrained=False, 
    ).image_encoder
    
    student_wrapper = OutputDistillationWrapper(
        student_model=student_encoder, 
        student_out_channels=256, 
        teacher_out_channels=1024
    ).to(DEVICE)

    optimizer = AdamW(student_wrapper.parameters(), lr=LEARNING_RATE)
    # criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda')

    total_steps = (len(train_dataloader) // ACCUMULATION_STEPS) * EPOCHS
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=LEARNING_RATE, 
        total_steps=total_steps, 
        pct_start=0.1,
    )

    # Lists to track metrics for plotting
    history_train_loss = []
    history_val_loss = []
    
    os.makedirs("./model", exist_ok=True)

    for epoch in range(EPOCHS):
        # Training Loop
        student_wrapper.train()
        print(f"\nStarting Epoch {epoch+1}/{EPOCHS}")
        epoch_train_loss = 0.0
        optimizer.zero_grad()
        
        train_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training")
        
        for idx, images in train_bar:
            images = images.to(DEVICE)
            
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    t_outputs = teacher_encoder(pixel_values=images)
                    t_feats = t_outputs.last_hidden_state 
                    
                    if len(t_feats.shape) == 3:
                        B, N, C = t_feats.shape
                        H = W = int(N ** 0.5)
                        
                        if H * H == N:
                            spatial_tokens = t_feats
                        else:
                            spatial_tokens = t_feats[:, 1:, :]
                            H = W = int((N - 1) ** 0.5)
                            
                        t_feats = spatial_tokens.transpose(1, 2).reshape(B, C, H, W)

            target_spatial_size = t_feats.shape[2:]

            with torch.amp.autocast('cuda'):
                s_feats = student_wrapper(images, target_spatial_size)
                cos_sim = F.cosine_similarity(s_feats, t_feats, dim=1)

                # Loss is 1 minus the mean similarity
                loss = (1.0 - cos_sim.mean()) / ACCUMULATION_STEPS
                
            scaler.scale(loss).backward()
            
            if (idx + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                
            current_loss = loss.item() * ACCUMULATION_STEPS
            epoch_train_loss += current_loss
            
            train_bar.set_postfix({'Cosine Similarity Loss': f"{current_loss:.4f}"})
            
        if len(train_dataloader) % ACCUMULATION_STEPS != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        avg_train_loss = epoch_train_loss / len(train_dataloader)
        history_train_loss.append(avg_train_loss)

        # Validation Loop
        student_wrapper.eval()
        epoch_val_loss = 0.0
        
        val_bar = tqdm(val_dataloader, desc="Validation")
        with torch.no_grad():
            for images in val_bar:
                images = images.to(DEVICE)
                
                with torch.amp.autocast('cuda'):
                    t_outputs = teacher_encoder(pixel_values=images)
                    t_feats = t_outputs.last_hidden_state 
                    
                    if len(t_feats.shape) == 3:
                        B, N, C = t_feats.shape
                        H = W = int(N ** 0.5)
                        if H * H == N:
                            spatial_tokens = t_feats
                        else:
                            spatial_tokens = t_feats[:, 1:, :]
                            H = W = int((N - 1) ** 0.5)
                        t_feats = spatial_tokens.transpose(1, 2).reshape(B, C, H, W)

                target_spatial_size = t_feats.shape[2:]

                with torch.amp.autocast('cuda'):
                    s_feats = student_wrapper(images, target_spatial_size)
                    v_cos_sim = F.cosine_similarity(s_feats, t_feats, dim=1)
                    
                    v_loss = 1.0 - v_cos_sim.mean()
                    
                epoch_val_loss += v_loss.item()
                val_bar.set_postfix({'Val Loss': f"{v_loss.item():.4f}"})
                
        avg_val_loss = epoch_val_loss / len(val_dataloader)
        history_val_loss.append(avg_val_loss)

        print(f"Epoch {epoch+1} Summary - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        save_dir = "checkpoints"
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, f"efficientvit_distill_ep{epoch+1}_fromScratch.pth")
        
        torch.save(student_wrapper.state_dict(), save_path)
        print(f"Checkpoint saved successfully to {save_path}")

    print("\nGenerating Loss Plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), history_train_loss, marker='o', linestyle='-', label='Train Loss', color='blue')
    plt.plot(range(1, EPOCHS + 1), history_val_loss, marker='s', linestyle='--', label='Validation Loss', color='orange')
    plt.title('Knowledge Distillation: Train vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.xticks(range(1, EPOCHS + 1))
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = "./model/loss_curve_fromScratch.png"
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Loss plot saved to {plot_path}")

if __name__ == "__main__":
    main()