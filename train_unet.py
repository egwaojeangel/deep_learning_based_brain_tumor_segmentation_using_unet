# train_unet_safe.py - Memory-efficient version + resumable + gradient clipping

import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import random
from tqdm import tqdm
from pathlib import Path

# ────────────────────────────────────────────────────────────────
# CONFIG - easy to change
# ────────────────────────────────────────────────────────────────
CHECKPOINT_DIR       = Path("checkpoints_brats_unet")
LATEST_CHECKPOINT    = CHECKPOINT_DIR / "latest_checkpoint.pth"
BEST_CHECKPOINT      = CHECKPOINT_DIR / "best_model.pth"

NUM_EPOCHS           = 10       # ← Start with 10; increase later if needed
IMG_SIZE             = 96
BATCH_SIZE           = 4        # ← Reduced from 8 to avoid OOM during training
LR                   = 1e-3
SAVE_EVERY_BATCHES   = 100      # still good balance

DATA_ROOT = r"C:\Users\egwao\Downloads\archive (5)\BraTS2021_Training_Data"

# ────────────────────────────────────────────────────────────────
# 1. Dataset Class (unchanged)
# ────────────────────────────────────────────────────────────────
class BraTSDataset(Dataset):
    def __init__(self, patient_paths, img_size=IMG_SIZE, augment=False):
        self.img_size = img_size
        self.patient_paths = patient_paths
        self.augment = augment
        self.slices = []
        self._prepare_slices()

    def _prepare_slices(self):
        for patient_path in self.patient_paths:
            flair_path = seg_path = None
            for file in os.listdir(patient_path):
                if "flair" in file.lower(): flair_path = os.path.join(patient_path, file)
                elif "seg" in file.lower(): seg_path = os.path.join(patient_path, file)
            if flair_path is None or seg_path is None:
                continue
            seg_img = nib.load(seg_path)
            seg_data = seg_img.get_fdata()
            for i in range(seg_data.shape[2]):
                if np.sum(seg_data[:, :, i]) > 0:
                    self.slices.append((patient_path, i))
        print(f"Total 2D slices with tumor: {len(self.slices)}")

    def __len__(self): return len(self.slices)

    def __getitem__(self, idx):
        patient_path, i = self.slices[idx]
        flair_path = seg_path = None
        for file in os.listdir(patient_path):
            if "flair" in file.lower(): flair_path = os.path.join(patient_path, file)
            elif "seg" in file.lower(): seg_path = os.path.join(patient_path, file)
        flair_img = nib.load(flair_path)
        seg_img = nib.load(seg_path)
        image = np.array(flair_img.dataobj[:, :, i], dtype=np.float32)
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
        mask = np.array(seg_img.dataobj[:, :, i], dtype=np.float32)
        mask = np.where(mask > 0, 1, 0).astype(np.float32)

        if self.augment:
            if random.random() > 0.5:
                image = np.fliplr(image); mask = np.fliplr(mask)
            if random.random() > 0.5:
                image = np.flipud(image); mask = np.flipud(mask)
            if random.random() > 0.6:
                k = random.randint(1, 3)
                image = np.rot90(image, k); mask = np.rot90(mask, k)

        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        return (torch.tensor(np.expand_dims(image, 0), dtype=torch.float32),
                torch.tensor(np.expand_dims(mask, 0), dtype=torch.float32))


# ────────────────────────────────────────────────────────────────
# 2. U-Net Model (unchanged)
# ────────────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        self.dconv_down1 = DoubleConv(in_ch, 64)
        self.dconv_down2 = DoubleConv(64, 128)
        self.dconv_down3 = DoubleConv(128, 256)
        self.dconv_down4 = DoubleConv(256, 512)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dconv_up3 = DoubleConv(256 + 512, 256)
        self.dconv_up2 = DoubleConv(128 + 256, 128)
        self.dconv_up1 = DoubleConv(128 + 64, 64)
        self.conv_last = nn.Conv2d(64, out_ch, 1)

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
# 3. Checkpoint functions (unchanged)
# ────────────────────────────────────────────────────────────────
def save_checkpoint(model, optimizer, epoch, val_dice, current_batch=-1, is_best=False):
    CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_dice': val_dice,
        'current_batch': current_batch,
    }
    torch.save(state, LATEST_CHECKPOINT)
    msg = f"epoch {epoch+1}"
    if current_batch >= 0: msg += f", batch {current_batch+1}"
    print(f"→ Saved latest checkpoint: {LATEST_CHECKPOINT} ({msg})")
    if is_best:
        torch.save(state, BEST_CHECKPOINT)
        print(f"→ Saved BEST model: {BEST_CHECKPOINT} (Dice: {val_dice:.4f})")


def load_checkpoint(model, optimizer):
    if not LATEST_CHECKPOINT.exists():
        print("No checkpoint found → starting from epoch 0, batch 0")
        return 0, -1.0, 0
    try:
        checkpoint = torch.load(LATEST_CHECKPOINT, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_dice = checkpoint.get('val_dice', -1.0)
        resume_batch = checkpoint.get('current_batch', 0)
        if resume_batch > 0:
            print(f"→ Resuming epoch {epoch+1} from batch {resume_batch+1}")
        else:
            epoch += 1
            resume_batch = 0
            print(f"→ Starting/Resuming from epoch {epoch+1}, batch 1")
        print(f"   Previous best Dice: {best_dice:.4f}")
        return epoch, best_dice, resume_batch
    except Exception as e:
        print(f"Checkpoint loading failed: {e}\nStarting from scratch.")
        return 0, -1.0, 0


# ────────────────────────────────────────────────────────────────
# 4. Main script
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    all_patients = [os.path.join(DATA_ROOT, f) for f in os.listdir(DATA_ROOT)
                    if os.path.isdir(os.path.join(DATA_ROOT, f))]
    random.shuffle(all_patients)
    n = len(all_patients)
    train_patients = all_patients[:int(0.70 * n)]
    val_patients   = all_patients[int(0.70 * n):int(0.85 * n)]
    test_patients  = all_patients[int(0.85 * n):]

    print(f"Training patients: {len(train_patients)}")
    print(f"Validation patients: {len(val_patients)}")
    print(f"Testing patients: {len(test_patients)}")

    train_dataset = BraTSDataset(train_patients, img_size=IMG_SIZE, augment=True)
    val_dataset   = BraTSDataset(val_patients,   img_size=IMG_SIZE, augment=False)
    test_dataset  = BraTSDataset(test_patients,  img_size=IMG_SIZE, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    device = torch.device("cpu")
    print("Using device: CPU")

    model = UNet(in_ch=1, out_ch=1).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    start_epoch, best_val_dice, resume_from_batch = load_checkpoint(model, optimizer)

    epoch_pbar = tqdm(range(start_epoch, NUM_EPOCHS), desc="Overall Progress", unit="epoch")

    for epoch in epoch_pbar:
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"{'-'*60}")

        # ─── Training ────────────────────────────────────────────────
        model.train()
        train_loss = train_inter = train_union = train_batches = 0.0

        train_loop = tqdm(train_loader, desc="Train", leave=False,
                          initial=resume_from_batch, total=len(train_loader))

        skipped = False
        for batch_idx, (images, masks) in enumerate(train_loop):
            if batch_idx < resume_from_batch: continue
            if not skipped and resume_from_batch > 0:
                print(f"   → Skipped first {resume_from_batch} batches")
                skipped = True

            try:
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # ← Added clipping
                optimizer.step()

                train_loss += loss.item()
                train_batches += 1

                preds = (outputs > 0.5).float()
                inter = (preds * masks).sum().item()
                union = preds.sum().item() + masks.sum().item()
                train_inter += inter
                train_union += union

                train_loop.set_postfix(loss=f"{loss.item():.4f}",
                                     dice=f"{(2*inter)/(union+1e-8):.4f}")

                if (batch_idx + 1) % SAVE_EVERY_BATCHES == 0:
                    print(f"  → Mid-epoch save at batch {batch_idx+1}/{len(train_loader)}")
                    save_checkpoint(model, optimizer, epoch, best_val_dice,
                                    current_batch=batch_idx, is_best=False)

            except Exception as e:
                print(f"Batch {batch_idx} failed: {e} — skipping")
                continue

        train_loss /= train_batches if train_batches > 0 else 1
        train_dice = (2 * train_inter) / (train_union + 1e-8)
        resume_from_batch = 0

        # ─── Validation (memory-efficient) ───────────────────────────
        model.eval()
        val_loss = val_inter = val_union = val_batches = 0.0
        tp = fp = tn = fn = 0.0

        val_loop = tqdm(val_loader, desc="Valid", leave=False)

        with torch.no_grad():
            for images, masks in val_loop:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                val_batches += 1

                preds = (outputs > 0.5).float()
                inter = (preds * masks).sum().item()
                union = preds.sum().item() + masks.sum().item()
                val_inter += inter
                val_union += union

                tp += ((preds == 1) & (masks == 1)).sum().item()
                fp += ((preds == 1) & (masks == 0)).sum().item()
                tn += ((preds == 0) & (masks == 0)).sum().item()
                fn += ((preds == 0) & (masks == 1)).sum().item()

                val_loop.set_postfix(loss=f"{loss.item():.4f}",
                                   dice=f"{(2*inter)/(union+1e-8):.4f}")

        val_loss /= val_batches if val_batches > 0 else 1
        val_dice = (2 * val_inter) / (val_union + 1e-8)

        total = tp + fp + tn + fn
        val_acc  = (tp + tn) / total if total > 0 else 0.0
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        val_f1   = 2 * prec * rec / (prec + rec + 1e-8)
        val_sens = rec
        val_spec = tn / (tn + fp + 1e-8)

        print(f"\nTrain → Loss: {train_loss:.4f} | Dice: {train_dice:.4f}")
        print(f"Valid → Loss: {val_loss:.4f} | Dice: {val_dice:.4f}")
        print(f"      → Acc:  {val_acc:.4f} | F1:   {val_f1:.4f}")
        print(f"      → Sens: {val_sens:.4f} | Spec: {val_spec:.4f}")

        is_best = val_dice > best_val_dice
        if is_best:
            best_val_dice = val_dice
            print(f"   → New best validation Dice: {best_val_dice:.4f}")

        save_checkpoint(model, optimizer, epoch, val_dice,
                        current_batch=-1, is_best=is_best)

        epoch_pbar.set_postfix(best_dice=f"{best_val_dice:.4f}")

    # ────────────────────────────────────────────────────────────────
    # Final Testing (also memory-efficient)
    # ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL TEST EVALUATION (using best saved model)")
    print(f"{'='*60}")

    if BEST_CHECKPOINT.exists():
        checkpoint = torch.load(BEST_CHECKPOINT, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("→ Loaded best model for testing")
    else:
        print("→ No best model found, using final model state")

    model.eval()
    test_inter = test_union = test_batches = 0.0
    tp = fp = tn = fn = 0.0

    test_loop = tqdm(test_loader, desc="Test")

    with torch.no_grad():
        for images, masks in test_loop:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = (outputs > 0.5).float()

            inter = (preds * masks).sum().item()
            union = preds.sum().item() + masks.sum().item()
            test_inter += inter
            test_union += union
            test_batches += 1

            tp += ((preds == 1) & (masks == 1)).sum().item()
            fp += ((preds == 1) & (masks == 0)).sum().item()
            tn += ((preds == 0) & (masks == 0)).sum().item()
            fn += ((preds == 0) & (masks == 1)).sum().item()

    test_dice = (2 * test_inter) / (test_union + 1e-8)
    total = tp + fp + tn + fn
    test_acc  = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    test_f1   = 2 * prec * rec / (prec + rec + 1e-8)
    test_sens = rec
    test_spec = tn / (tn + fp + 1e-8)

    print(f"\nTest Dice:        {test_dice:.4f}")
    print(f"Test Accuracy:    {test_acc:.4f}")
    print(f"Test F1 Score:    {test_f1:.4f}")
    print(f"Test Sensitivity: {test_sens:.4f}")
    print(f"Test Specificity: {test_spec:.4f}")

    print("\nDone! Checkpoints saved in:", CHECKPOINT_DIR.resolve())