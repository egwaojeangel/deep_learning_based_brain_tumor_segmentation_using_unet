from flask import Flask, request, jsonify, send_from_directory
import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import nibabel as nib
import torchvision.transforms as T
from pathlib import Path

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm', 'nii', 'nii.gz'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = r"C:\Users\egwao\checkpoints_brats_unet\best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 96

# ─── UNet Model (unchanged) ────────────────────────────────────────────────
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

model = UNet(in_ch=1, out_ch=1).to(DEVICE)

try:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"→ Model loaded successfully from {MODEL_PATH} on {DEVICE}")
except Exception as e:
    print(f"→ Model loading failed: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def segment_image(image_path):
    if model is None:
        raise ValueError("Model failed to load")

    ext = Path(image_path).suffix.lower()

    if ext in ['.nii', '.nii.gz']:
        nii = nib.load(image_path)
        data = nii.get_fdata()
        if len(data.shape) != 3:
            raise ValueError("NIfTI file must be 3D volume")

        # Choose middle slice for simplicity
        slice_idx = data.shape[2] // 2
        img_slice = data[:, :, slice_idx]
        img_slice = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice) + 1e-8)
        img_slice = np.clip(img_slice, 0, 1)
        orig_img = img_slice
        img = Image.fromarray((img_slice * 255).astype(np.uint8))

    else:
        img = Image.open(image_path).convert('L')
        orig_img = np.array(img) / 255.0
        orig_img = np.clip(orig_img, 0, 1)

    orig_h, orig_w = orig_img.shape[:2]

    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
    ])
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(tensor)
        mask = pred.squeeze().cpu().numpy()

    mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    orig_rgb = cv2.cvtColor((orig_img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB) \
        if ext in ['.nii', '.nii.gz'] else cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    tumor_mask = (mask > 0.3).astype(np.uint8) * 255
    overlay = orig_rgb.copy()
    overlay[tumor_mask > 0] = [255, 0, 0]  # red overlay

    result_filename = f"segmented_{Path(image_path).stem}.jpg"
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
    cv2.imwrite(result_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    confidence = float(np.max(mask)) * 100
    result_msg = "Positive (Tumor Segmented)" if confidence > 1.0 else "No tumor detected"

    return {
        'segmented_url': f"/uploads/{result_filename}",
        'confidence': round(confidence, 2),
        'result': result_msg,
        'dice_estimate': round(confidence / 100, 4)
    }

@app.route('/')
def index():
    return send_from_directory('.', 'tumorseg.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            result = segment_image(filepath)
            return jsonify({'success': True, **result})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)