# 🧠 Brain Tumor Segmentation Using U-Net on MRI Scans  

**Core Stack:** Python (PyTorch) · Deep Learning · Medical Image Segmentation · MRI  
**Model:** U-Net  
**Dataset:** BraTS 2021 (FLAIR modality)

---

## Overview  
This project presents a deep learning–based brain tumor segmentation system implemented using Magnetic Resonance Imaging (MRI) scans. The system is built around the **U-Net architecture**, a well-established convolutional neural network designed for biomedical image segmentation.

The implementation focuses on **pixel-level tumor segmentation** from 2D MRI slices extracted from the **BraTS 2021 dataset**, using the **FLAIR modality**. Rather than conducting a full clinical study, this project emphasizes **correct implementation, training stability, evaluation, and reproducibility** of a modern medical image segmentation pipeline.

Brain tumor segmentation plays a critical role in diagnosis, treatment planning, and disease monitoring. Manual delineation by radiologists is time-consuming and subject to inter-observer variability, motivating the use of automated deep learning–based solutions.

---

## How to Run Locally

### Prerequisites
- Python 3.10 or higher
- Git
- A trained model checkpoint (`best_model.pth`)

---

### 1. Clone the repository

git clone https://github.com/egwaojeangel/deep_learning_based_brain_tumor_segmentation_using_unet.git
cd deep_learning_based_brain_tumor_segmentation_using_unet

---

### 2. Create a virtual environment (recommended)

python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate

---

### 3. Install dependencies

pip install -r requirements.txt

---

### 4. Add your model checkpoint

Place your trained `best_model.pth` file in a folder called `checkpoints`:

deep_learning_based_brain_tumor_segmentation_using_unet/
├── checkpoints/
│   └── best_model.pth
├── brainsegapp.py
├── tumorseg.html
└── ...

Then open `brainsegapp.py` and update this line to match your path:

MODEL_PATH = "checkpoints/best_model.pth"

---

### 5. Run the app

python brainsegapp.py

Then open your browser and go to:

http://127.0.0.1:5000

---

### Supported Scan Formats
- `.jpg`, `.png` — standard brain scan images
- `.nii`, `.nii.gz` — NIfTI format (real hospital MRI files)

  ---

## Key Objectives  
- Implement a U-Net–based segmentation model for brain tumor detection  
- Train and evaluate the model on BraTS MRI data  
- Achieve reliable segmentation performance using Dice-based evaluation  
- Build a robust, checkpoint-resumable training pipeline  
- Demonstrate practical understanding of medical image preprocessing, augmentation, and evaluation  

---

## Results  
The trained model demonstrated strong segmentation performance on unseen test data:

- **Test Dice Score:** 91.47% 
- **Test Accuracy:** 99.58%  
- **Test F1 Score:** 92.16%  
- **Test Sensitivity (Recall):** 90.64%  
- **Test Specificity:** 99.83%

  ![Test Results](https://github.com/egwaojeangel/deep_learning_based_brain_tumor_segmentation_using_unet/blob/main/screenshots_on_tumor_seg/brain_segmentation_test_results.png)

Checkpoints saved in:  
`C:\Users\egwao\checkpoints_brats_unet`

These results indicate high overlap between predicted tumor regions and ground-truth masks, with excellent background classification and minimal false positives.

---

## Dataset  
This project uses the **BraTS 2021 (Brain Tumor Segmentation Challenge)** dataset, a widely adopted benchmark in medical imaging research.

### Dataset Characteristics  
- Multi-institutional MRI scans  
- Expert-annotated tumor segmentation masks  
- Multiple MRI modalities (FLAIR, T1, T1ce, T2)  

### Modality Used in This Project  
**FLAIR (Fluid-Attenuated Inversion Recovery)**  
Chosen due to its strong contrast for highlighting tumor-associated edema.

> ⚠️ **Note:**  
> Due to size and licensing restrictions, the BraTS dataset is not included in this repository.

---

## Data Preparation & Split  
Patient-level split to prevent data leakage  
- Training: 70%  
- Validation: 15%  
- Testing: 15%  

Only **2D slices containing tumor regions** were selected for training and evaluation, improving class balance and segmentation relevance.

---

## Methodology 
### Image Preprocessing  
- Slice extraction from 3D NIfTI volumes  
- Min–max intensity normalization  
- Resizing to **96 × 96 pixels**  
- Binary mask generation (tumor vs. background)  

Each MRI slice is treated as a **single-channel (grayscale)** input.

### Data Augmentation (Training Only)  
To improve generalization, the following augmentations were applied:

- Horizontal flipping  
- Vertical flipping  
- Random 90° rotations  

Augmentations were applied consistently to both images and masks to preserve spatial alignment.

---

## Model Architecture  
### U-Net  
The model is based on the original **U-Net architecture**, consisting of:

**Encoder (Contracting Path):**  
- Stacked convolutional blocks  
- Batch normalization  
- ReLU activations  
- Max-pooling for spatial downsampling  

**Decoder (Expanding Path):**  
- Bilinear upsampling  
- Skip connections from encoder layers  
- Feature concatenation for spatial detail recovery  

**Output:**  
- 1×1 convolution  
- Sigmoid activation  
- Binary segmentation mask (tumor / non-tumor)  

---

## Training Details  
- **Framework:** PyTorch  
- **Loss Function:** Binary Cross-Entropy (BCELoss)  
- **Optimizer:** Adam  
- **Learning Rate:** 1e-3  
- **Batch Size:** 4  
- **Epochs:** 10  
- **Device:** CPU  

### Training Stability Techniques  
- Gradient clipping (max norm = 1.0)  
- Mid-epoch checkpoint saving  
- Automatic resume from last checkpoint  
- Best-model tracking based on validation Dice score  

---

## Evaluation Metrics  
The model was evaluated using standard segmentation metrics:

- Dice Coefficient (primary metric)  
- Accuracy  
- Precision  
- Recall (Sensitivity)  
- Specificity  
- F1 Score  

Dice score was prioritized due to its robustness for class-imbalanced segmentation tasks.

---

## Checkpointing & Reproducibility  
The training pipeline includes:

- `latest_checkpoint.pth` – latest training state  
- `best_model.pth` – best model based on validation Dice score  

Training can be safely resumed after interruption without loss of progress.

---

## Relation to Existing Research  
U-Net–based architectures remain a dominant baseline for brain tumor segmentation tasks. Recent studies continue to demonstrate their effectiveness on BraTS datasets, particularly when combined with careful preprocessing and evaluation strategies (Isensee et al., 2021; Hatamizadeh et al., 2022; Zhang et al., 2023).

This project aligns with these works by implementing a clean, reproducible U-Net pipeline, serving as a strong baseline for future extensions.

---

## Limitations  
- Uses 2D slice-based segmentation instead of full 3D volumes  
- Single MRI modality (FLAIR only)  
- No clinical validation or expert radiologist review  
- No explainability or uncertainty estimation  

---

## Future Work  
- Extend to 3D U-Net or nnU-Net  
- Incorporate multi-modal MRI inputs  
- Add Dice + Focal loss combinations  
- Integrate explainability methods  
- Deploy as a clinical decision support prototype  
- Validate with expert radiologist annotations  

---

## Disclaimer  
This project is intended **strictly for research and educational purposes**.  
It is not a certified medical device and must not be used for clinical diagnosis or treatment decisions.

---

## Sample Test Scans

You can test the app immediately using these sample MRI scans 
included in the repository:

Sample_MRI_Images/
├── 00000534_brain_flair.nii
├── braintumor.jpg


Simply upload any of these in the web app to see the 
segmentation in action.

---
### Demo Video (Click to Watch)
Watch the brain tumor segmentation model in action:

[![Watch the video](https://github.com/egwaojeangel/deep_learning_based_brain_tumor_segmentation_using_unet/blob/main/screenshots_on_tumor_seg/Upload_Scan.jpeg)](https://drive.google.com/file/d/1H175DdmRg-RpnKBPS9IlPN6SOq7HavQL/view?usp=drivesdk)

---

## Author  
**Angel Egwaoje**

