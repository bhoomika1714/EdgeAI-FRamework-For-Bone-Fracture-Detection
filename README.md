

## 🦴 Edge-AI Fracture Detection and Segmentation using MobileNetV3 + U-Net

### Abstract

This project proposes a lightweight and edge-deployable **dual-stage deep learning framework** for **fracture classification and segmentation** from X-ray images. It leverages **MobileNetV3** for efficient fracture detection and **U-Net** for precise anatomical segmentation, optimized for mobile and embedded deployment using **Quantization-Aware Training (QAT)** and **ARM-layer fusion**.

---

###  Features

* 🧠 **Dual-stage architecture**: MobileNetV3 for classification, U-Net for segmentation
* ⚡ **Quantization-aware training (QAT)** for INT8 deployment
* 🔄 **PyTorch to TensorFlow Lite conversion**
* 🤖 **Android-compatible Flutter deployment**
* ⏱️ **Low-latency real-time inference** on ARM Cortex-A72 (62ms)
* 📊 **High accuracy**: 85.22% classification, 0.87 Dice score for segmentation
* 📦 Optimized for <25MB model size

---

###  Architecture

```
graph TD
    A[Input X-ray Image] --> B[MobileNetV3 Classifier]
    B -->|Fracture Detected| C[U-Net Segmenter]
    C --> D[Quantization + Layer Fusion]
    D --> E[TensorFlow Lite Conversion]
    E --> F[Mobile App (Flutter)]
```

---

### 🛠️ Tech Stack

| Component          | Tool/Library                                            |
| ------------------ | ------------------------------------------------------- |
| Classifier         | MobileNetV3 (with QAT)                                  |
| Segmenter          | U-Net                                                   |
| Training Framework | PyTorch                                                 |
| Deployment         | TensorFlow Lite + Flutter                               |
| Optimizations      | Quantization, ARM Layer Fusion                          |
| Dataset            | [FracAtlas](https://doi.org/10.1038/s41597-023-02432-4) |

---

###  Performance Overview

| Metric                    | Pre-Quantization | Post-Quantization |
| ------------------------- | ---------------- | ----------------- |
| Accuracy                  | 89.84%           | 85.22%            |
| Dice Score (Segmentation) | 0.87             | 0.86              |
| IoU (Segmentation)        | 0.79             | 0.78              |
| Inference Latency (ms)    | 145              | 62                |
| Model Size (MB)           | 85               | 25                |

---

###  Dataset: FracAtlas

* Total images: 4,083 X-rays
* Regions covered: hand, leg, hip, shoulder
* Split:

  * Train: 3,000 images
  * Val: 600 images
  * Test: 483 images

---

### 📁 Project Structure

```
├── models/
│   ├── mobilenetv3_classifier.pt
│   └── unet_segmenter.pt
├── tflite_models/
│   ├── mobilenetv3_q.tflite
│   └── unet_q.tflite
├── android_app/
│   └── flutter_project/
├── scripts/
│   ├── train_classifier.py
│   ├── train_segmenter.py
│   └── convert_to_tflite.py
├── dataset/
│   └── FracAtlas/
├── README.md
└── requirements.txt
```

---

### 🚀 Getting Started

#### Step 1: Clone and Setup

```bash
git clone https://github.com/yourusername/edge-ai-fracture-detection.git
cd edge-ai-fracture-detection
pip install -r requirements.txt
```

#### Step 2: Train or Download Models

```bash
# Train classifier and segmenter or use pre-trained models
python scripts/train_classifier.py
python scripts/train_segmenter.py
```

#### Step 3: Convert to TFLite

```bash
python scripts/convert_to_tflite.py
```

#### Step 4: Deploy on Android

* Open `android_app/flutter_project` in Android Studio
* Add `.tflite` files to assets
* Build and run the app

---

### 📊 Training Details

| Parameter      | Value           |
| -------------- | --------------- |
| Image Size     | 256 × 256       |
| Optimizer      | Adam            |
| Learning Rate  | 1e-4            |
| Epochs         | 70              |
| Loss Functions | BCE + Dice Loss |


### 📌 Future Work

* 🧒 Pediatric fracture support
* 🧠 Transformer or attention-based architecture
* 🏥 Clinical deployment & PACS integration
* 🧬 3D CT/MRI data support
* 🔧 Model pruning & mixed-precision inference

---

### 👩‍💻 Contributors

* Shruti Sutar
* **Bhoomika Marigoudar**
* Saakshi Lokhande
* Shribhakti S Vibhuti
* Snehal V Devasthale
* Dr. Uday Kulkarni *(Supervisor)*

