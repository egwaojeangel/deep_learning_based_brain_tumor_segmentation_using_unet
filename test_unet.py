

import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
from tqdm import tqdm
from pathlib import Path

# ────────────────────────────────────────────────────────────────
# CONFIG (should match training)
# ────────────────────────────────────────────────────────────────
IMG_SIZE   = 96
BATCH_SIZE = 4

DATA_ROOT       = r"C:\Users\egwao\Downloads\archive (5)\BraTS2021_Training_Data"
CHECKPOINT_DIR  = Path("checkpoints_brats_unet")
BEST_CHECKPOINT = CHECKPOINT_DIR / "best_model.pth"

# ────────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────────
class BraTSDataset(Dataset):
    def __init__(self, patient_paths, img_size=IMG_SIZE):
        self.img_size = img_size
        self.patient_paths = patient_paths
        self.slices = []
        self._prepare_slices()

    def _prepare_slices(self):
        for patient_path in self.patient_paths:
            flair_path = seg_path = None
            for file in os.listdir(patient_path):
                fname = file.lower()
                if "flair" in fname:
                    flair_path = os.path.join(patient_path, file)
                elif "seg" in fname:
                    seg_path = os.path.join(patient_path, file)
            
            if flair_path is None or seg_path is None:
                continue
                
            seg_data = nib.load(seg_path).get_fdata()
            for slice_idx in range(seg_data.shape[2]):
                if np.sum(seg_data[:, :, slice_idx]) > 0:
                    self.slices.append((patient_path, slice_idx))

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        patient_path, slice_idx = self.slices[idx]
        
        flair_path = seg_path = None
        for file in os.listdir(patient_path):
            fname = file.lower()
            if "flair" in fname:
                flair_path = os.path.join(patient_path, file)
            elif "seg" in fname:
                seg_path = os.path.join(patient_path, file)

        image = np.array(nib.load(flair_path).dataobj[:, :, slice_idx], dtype=np.float32)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        mask = np.array(nib.load(seg_path).dataobj[:, :, slice_idx], dtype=np.float32)
        mask = np.where(mask > 0, 1.0, 0.0).astype(np.float32)

        image = cv2.resize(image, (self.img_size, self.img_size))
        mask  = cv2.resize(mask,  (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        return (
            torch.from_numpy(image).unsqueeze(0).float(),
            torch.from_numpy(mask).unsqueeze(0).float()
        )

# ────────────────────────────────────────────────────────────────
# U-Net Model (must match training exactly)
# ────────────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()

        # Encoder
        self.dconv_down1 = DoubleConv(in_ch,   64)
        self.dconv_down2 = DoubleConv(64,  128)
        self.dconv_down3 = DoubleConv(128, 256)
        self.dconv_down4 = DoubleConv(256, 512)

        self.maxpool = nn.MaxPool2d(2)

        # Decoder
        self.upsample   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.dconv_up3 = DoubleConv(512 + 256, 256)
        self.dconv_up2 = DoubleConv(256 + 128, 128)
        self.dconv_up1 = DoubleConv(128 + 64,   64)

        self.conv_last = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        conv2 = self.dconv_down2(self.maxpool(conv1))
        conv3 = self.dconv_down3(self.maxpool(conv2))
        conv4 = self.dconv_down4(self.maxpool(conv3))

        x = self.upsample(conv4)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)

        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)

        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)

        return torch.sigmoid(self.conv_last(x))


# ────────────────────────────────────────────────────────────────
# Main - Test Evaluation
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    print("═"*75)
    print("FINAL TEST EVALUATION  (Best Saved Model)")
    print("═"*75)

    if not BEST_CHECKPOINT.exists():
        print(f"ERROR: Best model not found → {BEST_CHECKPOINT}")
        exit(1)

    # Patient split (same 85/15 logic as training)
    all_patients = [
        os.path.join(DATA_ROOT, p) 
        for p in os.listdir(DATA_ROOT) 
        if os.path.isdir(os.path.join(DATA_ROOT, p))
    ]

    test_start_idx = int(0.85 * len(all_patients))
    test_patients = all_patients[test_start_idx:]

    print(f"Test patients: {len(test_patients)}   |   Total slices depend on tumor presence\n")

    test_dataset = BraTSDataset(test_patients, img_size=IMG_SIZE)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # Load model
    model = UNet(in_ch=1, out_ch=1).to(device)
    checkpoint = torch.load(BEST_CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("→ Loaded best model successfully\n")

    # ─── Evaluation ──────────────────────────────────────────────────
    tp = fp = tn = fn = 0.0
    inter = union = 0.0

    with torch.no_grad():
        for imgs, masks in tqdm(test_loader, desc="Evaluating test set"):
            imgs  = imgs.to(device)
            masks = masks.to(device)

            preds = (model(imgs) > 0.5).float()

            inter += (preds * masks).sum().item()
            union += preds.sum().item() + masks.sum().item()

            tp += ((preds == 1) & (masks == 1)).sum().item()
            fp += ((preds == 1) & (masks == 0)).sum().item()
            tn += ((preds == 0) & (masks == 0)).sum().item()
            fn += ((preds == 0) & (masks == 1)).sum().item()

    # Final metrics
    dice        = 2 * inter / (union + 1e-8)
    accuracy    = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    sensitivity = tp / (tp + fn + 1e-8)           # = Recall
    specificity = tn / (tn + fp + 1e-8)
    f1          = 2 * tp / (2 * tp + fp + fn + 1e-8)

    total_pixels = tp + tn + fp + fn

    print("\n" + "─"*75)
    print("FINAL TEST RESULTS")
    print("─"*75)
    print(f"Test Dice Score    : {dice:.4f}")
    print(f"Test Accuracy      : {accuracy:.4f}")
    print(f"Test F1 Score      : {f1:.4f}")
    print(f"Test Sensitivity   : {sensitivity:.4f}    (Recall)")
    print(f"Test Specificity   : {specificity:.4f}")
    print()
    print("Confusion Matrix (pixel counts):")
    print(f"  True Positive  (TP) : {tp:,.0f}")
    print(f"  False Positive (FP) : {fp:,.0f}")
    print(f"  True Negative  (TN) : {tn:,.0f}")
    print(f"  False Negative (FN) : {fn:,.0f}")
    print(f"  Total pixels evaluated : {total_pixels:,.0f}")
    print("─"*75)

    print("\nEvaluation finished.")