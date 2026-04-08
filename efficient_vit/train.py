import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


class SA1B_Dataset(Dataset):
    """Scans a directory for images"""
    def __init__(self, data_dir, image_size=1024):
        self.data_dir = data_dir
        self.image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB') 
        return self.transform(image)


class HookFeatureExtractor(nn.Module):
    """Secretly captures intermediate feature maps without altering source code."""
    def __init__(self, model, target_layer_names):
        super().__init__()
        self.model = model
        self.target_layer_names = target_layer_names
        self.features = {}

        for name, layer in self.model.named_modules():
            if name in self.target_layer_names:
                layer.register_forward_hook(self.save_outputs_hook(name))

    def save_outputs_hook(self, layer_name):
        def fn(_, __, output):
            if isinstance(output, tuple):
                output = output[0]
            self.features[layer_name] = output
        return fn

    def forward(self, x):
        self.features.clear()
        _ = self.model(x) 
        return [self.features[name] for name in self.target_layer_names]


class DynamicDistillationWrapper(nn.Module):
    """Dynamically builds 1x1 convolutions to match Student channels to Teacher channels."""
    def __init__(self, student_extractor, teacher_extractor, image_size=1024, device='cuda'):
        super().__init__()
        self.student = student_extractor
        self.teacher = teacher_extractor
        self.projections = nn.ModuleList()
        self._build_projections(image_size, device)

    def _build_projections(self, image_size, device):
        self.student.to(device)
        self.teacher.to(device)
        
        dummy_input = torch.randn(1, 3, image_size, image_size).to(device)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                s_feats = self.student(dummy_input)
                t_feats = self.teacher(dummy_input)
        
        for s_feat, t_feat in zip(s_feats, t_feats):
            s_channels = s_feat.shape[1]
            t_channels = t_feat.shape[1] if len(t_feat.shape) == 4 else t_feat.shape[-1]
            self.projections.append(nn.Conv2d(s_channels, t_channels, kernel_size=1).to(device))

    def forward(self, x):
        student_features = self.student(x)
        return [proj(feat) for proj, feat in zip(self.projections, student_features)]


class SpatialMatchingLoss(nn.Module):
    """Interpolates student spatial dims to match teacher, then calculates MSE."""
    def __init__(self):
        super().__init__()

    def forward(self, student_feats, teacher_feats):
        total_loss = 0.0
        
        for s_feat, t_feat in zip(student_feats, teacher_feats):
            if len(t_feat.shape) == 3: 
                B, N, C = t_feat.shape
                H = W = int(N ** 0.5) 
                t_feat = t_feat.transpose(1, 2).reshape(B, C, H, W)
            
            if s_feat.shape[2:] != t_feat.shape[2:]:
                s_feat = F.interpolate(
                    s_feat, size=t_feat.shape[2:], mode='bilinear', align_corners=False
                )
            
            total_loss += F.mse_loss(s_feat, t_feat)
        return total_loss


def main():
    # --- Configuration ---
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPOCHS = 1
    ACCUMULATION_STEPS = 4
    LEARNING_RATE = 1e-4
    IMAGE_SIZE = 1008 
    BATCH_SIZE = 4 
    DATA_DIR = "/content/data"
    MODEL_DIR = "/content/drive/MyDrive/SAM3_Distillation/model"

    # --- Setup Local Data ---
    print("Initializing local Dataloader from ../data ...")
    dataset = SA1B_Dataset(data_dir=DATA_DIR, image_size=IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # --- Setup Models ---
    print("Loading models...")
    
    # 1. Load SAM3
    from transformers import Sam3Model
    
    sam3_model = Sam3Model.from_pretrained(MODEL_DIR)
    sam3_model.to(DEVICE)
    sam3_model.eval()

    # 2. Load EfficientViT
    from efficientvit.sam_model_zoo import create_efficientvit_sam_model
    efficientvit_student = create_efficientvit_sam_model(name="efficientvit-sam-l0", pretrained=False).image_encoder

    # Define target layers to align
    student_layers = ['backbone.stages.0', 'backbone.stages.1', 'backbone.stages.2', 'backbone.stages.3']
    teacher_layers = ['backbone.layers.7', 'backbone.layers.15', 'backbone.layers.23', 'backbone.layers.31'] # Adjust to match SAM3 block config
    
    # Wrap models
    teacher_extractor = HookFeatureExtractor(sam3_model.vision_encoder, teacher_layers).eval()
    student_extractor = HookFeatureExtractor(efficientvit_student, student_layers)
    distillation_model = DynamicDistillationWrapper(student_extractor, teacher_extractor, IMAGE_SIZE, DEVICE)
    
    # --- Setup Training Components (CPU Friendly) ---
    optimizer = AdamW(distillation_model.parameters(), lr=LEARNING_RATE)
    criterion = SpatialMatchingLoss()
    scaler = torch.amp.GradScaler('cuda')

    # --- Training Loop ---
    for epoch in range(EPOCHS):
        distillation_model.train()
        print(f"\nStarting Epoch {epoch+1}/{EPOCHS}")
        
        epoch_loss = 0.0
        optimizer.zero_grad()
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        
        for idx, images in progress_bar:
            images = images.to(DEVICE)
            
            # 1. Teacher Pass
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    t_feats = teacher_extractor(images)
                    
            # 2. Student Pass & Loss
            with torch.amp.autocast('cuda'):
                s_feats = distillation_model(images)
                loss = criterion(s_feats, t_feats) / ACCUMULATION_STEPS
                
            # 3. Backward & Step
            scaler.scale(loss).backward()
            
            if (idx + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            # Metrics & Cleanup
            current_loss = loss.item() * ACCUMULATION_STEPS
            epoch_loss += current_loss
            progress_bar.set_postfix({'Loss': f"{current_loss:.4f}"})
            
            del images, t_feats, s_feats, loss
        
        # End of epoch cleanup
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}")
        
        torch.cuda.empty_cache()
        gc.collect()

        # Save checkpoint to the /model directory
        save_path = f"/model/efficientvit_l0_sam3_ep{epoch+1}.pth"
        torch.save(distillation_model.student.model.state_dict(), save_path)
        print(f"Checkpoint saved to {save_path}")

if __name__ == "__main__":
    main()
    pass