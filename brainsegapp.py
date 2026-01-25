from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sqlite3
import os
from datetime import datetime
from werkzeug.utils import secure_filename
import numpy as np
import nibabel as nib
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from pathlib import Path

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'dcm', 'nii', 'nii.gz'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DB_PATH = 'patients.db'
MODEL_PATH = r"C:\Users\egwao\checkpoints_brats_unet\best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 96

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
        self.dconv_up3 = DoubleConv(512 + 256, 256)
        self.dconv_up2 = DoubleConv(256 + 128, 128)
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
    print(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"Model loading failed: {e}")
    model = None

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id TEXT PRIMARY KEY,
            full_name TEXT NOT NULL,
            dob TEXT,
            gender TEXT,
            phone TEXT,
            address TEXT,
            emergency_contact TEXT,
            insurance TEXT,
            assigned_doctor TEXT,
            family_history_brain TEXT,
            symptoms TEXT,
            previous_conditions TEXT,
            current_medications TEXT,
            scan_date TEXT,
            scan_type TEXT,
            tumor_segmented_by_ai TEXT,
            tumor_size TEXT,
            tumor_type TEXT,
            tumor_location TEXT,
            tumor_grade TEXT,
            biopsy_result TEXT,
            histology TEXT,
            diagnosis_date TEXT,
            treatment_type TEXT,
            treatment_start TEXT,
            response_to_treatment TEXT,
            passport_url TEXT,
            created_at TEXT,
            updated_at TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            date TEXT,
            result TEXT,
            confidence REAL,
            tumor_size TEXT,
            tumor_type TEXT,
            tumor_location TEXT,
            tumor_grade TEXT,
            segmented TEXT,
            FOREIGN KEY(patient_id) REFERENCES patients(id)
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def segment_image(image_path, patient_id=None):
    if model is None:
        raise ValueError("Model not loaded")
    ext = Path(image_path).suffix.lower()
    try:
        if ext in ['.nii', '.nii.gz']:
            nii = nib.load(image_path)
            data = nii.get_fdata()
            if len(data.shape) != 3:
                raise ValueError(f"Expected 3D NIfTI, got {data.shape}")
            slice_idx = data.shape[2] // 2
            img_slice = data[:, :, slice_idx]
            img_slice = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice) + 1e-8)
            img_slice = np.clip(img_slice, 0, 1)
            orig_img = img_slice.copy()
            pil_img = Image.fromarray((img_slice * 255).astype(np.uint8))
        else:
            pil_img = Image.open(image_path).convert('L')
            orig_img = np.array(pil_img) / 255.0
            orig_img = np.clip(orig_img, 0, 1)

        orig_h, orig_w = orig_img.shape

        transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
        ])
        tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred = model(tensor)
            mask = pred.squeeze().cpu().numpy()

        mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        mask_uint8 = (mask_resized * 255).astype(np.uint8)
        _, tumor_mask_bin = cv2.threshold(mask_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        tumor_mask = (tumor_mask_bin > 0).astype(np.float32)

        if np.any(tumor_mask):
            confidence = float(np.mean(mask_resized[tumor_mask > 0])) * 100
        else:
            confidence = 0.0

        result_msg = "Positive (Tumor Segmented)" if confidence > 35 else "No significant tumor detected"

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(tumor_mask_bin, connectivity=8)
        if num_labels > 1:
            largest = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            area = stats[largest, cv2.CC_STAT_AREA]
            tumor_size_mm = round(np.sqrt(area) * 0.8, 1)
            cx, cy = centroids[largest]
            loc_x = "left" if cx < orig_w * 0.4 else "right" if cx > orig_w * 0.6 else "central"
            loc_y = "superior" if cy < orig_h * 0.4 else "inferior" if cy > orig_h * 0.6 else "middle"
            tumor_location = f"Approx {loc_x}-{loc_y}"
        else:
            tumor_size_mm = "N/A"
            tumor_location = "No significant tumor region"

        tumor_type = "Suspected glioma" if confidence > 50 else "Uncertain"
        tumor_grade = "AI estimate: Grade II–IV (biopsy required)"

        orig_gray_uint8 = (orig_img * 255).astype(np.uint8)
        orig_rgb = cv2.cvtColor(orig_gray_uint8, cv2.COLOR_GRAY2RGB)
        overlay = orig_rgb.copy()
        red_overlay = np.zeros_like(overlay)
        red_overlay[tumor_mask > 0] = [255, 0, 0]
        cv2.addWeighted(red_overlay, 0.45, overlay, 0.55, 0, overlay)

        segmented_filename = f"seg_{patient_id or 'unknown'}_{int(datetime.now().timestamp())}.jpg"
        segmented_path = os.path.join(app.config['UPLOAD_FOLDER'], segmented_filename)
        cv2.imwrite(segmented_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        return {
            'success': True,
            'result': result_msg,
            'confidence': round(confidence, 2),
            'tumor_size': f"{tumor_size_mm} mm" if tumor_size_mm != "N/A" else "N/A",
            'tumor_type': tumor_type,
            'tumor_location': tumor_location,
            'tumor_grade': tumor_grade,
            'segmented_url': f'/uploads/{segmented_filename}',
            'segmented_filename': segmented_filename
        }
    except Exception as e:
        raise RuntimeError(str(e))

@app.route('/')
def index():
    return send_from_directory('.', 'tumorseg.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/patients', methods=['GET', 'POST'])
def handle_patients():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if request.method == 'POST':
        data = request.form
        patient_id = data.get('id')
        now = datetime.now().isoformat()

        # Check if this is an update (patient already exists)
        c.execute('SELECT passport_url FROM patients WHERE id = ?', (patient_id,))
        existing = c.fetchone()
        existing_passport = existing[0] if existing else None

        passport_filename = existing_passport  # keep existing by default

        # If a new passport file was uploaded, replace it
        if 'passport' in request.files:
            passport_file = request.files['passport']
            if passport_file and allowed_file(passport_file.filename):
                passport_filename = secure_filename(passport_file.filename)
                passport_path = os.path.join(app.config['UPLOAD_FOLDER'], passport_filename)
                passport_file.save(passport_path)

        try:
            c.execute('''
                INSERT OR REPLACE INTO patients
                (id, full_name, dob, gender, phone, address, emergency_contact, insurance, assigned_doctor,
                 family_history_brain, symptoms, previous_conditions, current_medications, scan_date, scan_type,
                 tumor_segmented_by_ai, tumor_size, tumor_type, tumor_location, tumor_grade, biopsy_result,
                 histology, diagnosis_date, treatment_type, treatment_start, response_to_treatment, passport_url,
                 created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                patient_id,
                data.get('full_name', ''),
                data.get('dob'),
                data.get('gender'),
                data.get('phone'),
                data.get('address'),
                data.get('emergency_contact'),
                data.get('insurance'),
                data.get('assigned_doctor'),
                data.get('family_history_brain'),
                data.get('symptoms'),
                data.get('previous_conditions'),
                data.get('current_medications'),
                data.get('scan_date'),
                data.get('scan_type'),
                data.get('tumor_segmented_by_ai'),
                data.get('tumor_size'),
                data.get('tumor_type'),
                data.get('tumor_location'),
                data.get('tumor_grade'),
                data.get('biopsy_result'),
                data.get('histology'),
                data.get('diagnosis_date'),
                data.get('treatment_type'),
                data.get('treatment_start'),
                data.get('response_to_treatment'),
                passport_filename,
                now,  # created_at
                now   # updated_at
            ))
            conn.commit()
            return jsonify({'success': True})
        except Exception as e:
            conn.rollback()
            return jsonify({'error': str(e)}), 400
        finally:
            conn.close()

    # GET: list all patients
    c.execute('SELECT * FROM patients')
    patients = []
    for row in c.fetchall():
        passport_url = f'/uploads/{row[26]}' if row[26] else None
        patients.append({
            'id': row[0],
            'full_name': row[1],
            'dob': row[2],
            'gender': row[3],
            'phone': row[4],
            'address': row[5],
            'emergency_contact': row[6],
            'insurance': row[7],
            'assigned_doctor': row[8],
            'family_history_brain': row[9],
            'symptoms': row[10],
            'previous_conditions': row[11],
            'current_medications': row[12],
            'scan_date': row[13],
            'scan_type': row[14],
            'tumor_segmented_by_ai': row[15],
            'tumor_size': row[16],
            'tumor_type': row[17],
            'tumor_location': row[18],
            'tumor_grade': row[19],
            'biopsy_result': row[20],
            'histology': row[21],
            'diagnosis_date': row[22],
            'treatment_type': row[23],
            'treatment_start': row[24],
            'response_to_treatment': row[25],
            'passport_url': passport_url,
            'created_at': row[27],
            'updated_at': row[28]
        })
    conn.close()
    return jsonify(patients)

@app.route('/api/patient/<patient_id>', methods=['GET'])
def get_patient(patient_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT * FROM patients WHERE id = ?', (patient_id,))
    row = c.fetchone()
    conn.close()
    if row:
        passport_url = f'/uploads/{row[26]}' if row[26] else None
        return jsonify({
            'id': row[0],
            'full_name': row[1],
            'dob': row[2],
            'gender': row[3],
            'phone': row[4],
            'address': row[5],
            'emergency_contact': row[6],
            'insurance': row[7],
            'assigned_doctor': row[8],
            'family_history_brain': row[9],
            'symptoms': row[10],
            'previous_conditions': row[11],
            'current_medications': row[12],
            'scan_date': row[13],
            'scan_type': row[14],
            'tumor_segmented_by_ai': row[15],
            'tumor_size': row[16],
            'tumor_type': row[17],
            'tumor_location': row[18],
            'tumor_grade': row[19],
            'biopsy_result': row[20],
            'histology': row[21],
            'diagnosis_date': row[22],
            'treatment_type': row[23],
            'treatment_start': row[24],
            'response_to_treatment': row[25],
            'passport_url': passport_url,
            'updated_at': row[28]
        })
    return jsonify({'error': 'Patient not found'}), 404

@app.route('/api/patient/<patient_id>', methods=['DELETE'])
def delete_patient(patient_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM patients WHERE id = ?', (patient_id,))
    c.execute('DELETE FROM scans WHERE patient_id = ?', (patient_id,))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

@app.route('/api/patient/<patient_id>/scans', methods=['GET'])
def get_patient_scans(patient_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT date, result, confidence, tumor_size, tumor_type, tumor_location, tumor_grade, segmented FROM scans WHERE patient_id = ? ORDER BY date DESC', (patient_id,))
    scans = [{
        'date': row[0],
        'result': row[1],
        'confidence': row[2],
        'tumor_size': row[3],
        'tumor_type': row[4],
        'tumor_location': row[5],
        'tumor_grade': row[6],
        'segmented': f"/uploads/{row[7]}" if row[7] else None
    } for row in c.fetchall()]
    conn.close()
    return jsonify(scans)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'}), 400
    file = request.files['file']
    patient_id = request.form.get('patient_id')
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        try:
            result = segment_image(filepath, patient_id)
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute('''
                INSERT INTO scans (patient_id, date, result, confidence, tumor_size, tumor_type, tumor_location, tumor_grade, segmented)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                patient_id,
                datetime.now().isoformat(),
                result['result'],
                result['confidence'],
                result['tumor_size'],
                result['tumor_type'],
                result['tumor_location'],
                result['tumor_grade'],
                result['segmented_filename']
            ))
            conn.commit()
            conn.close()
            return jsonify(result)
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    return jsonify({'success': False, 'error': 'Invalid file'}), 400

if __name__ == '__main__':
    app.run(debug=True)