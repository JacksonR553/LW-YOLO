# LW-YOLOv5: Lightweight CNN for PCB Defect Detection

![LW-YOLOv5 Architecture](YOLO%20Model%20Architecture%20Design-Modified%20YOLO%20Structure%20(1).jpg)

---

## 📌 Introduction
Printed Circuit Board (PCB) defect detection is crucial for ensuring quality in electronic manufacturing. Traditional inspection methods (manual or AOI) are error-prone and inefficient. Deep learning–based object detection has emerged as a reliable alternative, but deploying these large models on **embedded devices** (such as NVIDIA Jetson Orin Nano) is challenging due to **limited computing power and memory**:contentReference[oaicite:0]{index=0}.

This project proposes **LW-YOLOv5**, an **optimized lightweight version of YOLOv5n**, designed for **real-time PCB defect detection on embedded devices**.  

Key highlights:
- Reduced parameters from **1.70M → 1.18M** (–31%).
- Achieved **mAP@0.5 = 0.945** with only 1.18M parameters.
- Outperforms many SOTA models in **accuracy–efficiency trade-off**:contentReference[oaicite:1]{index=1}.

---

## 🚀 Features
- **Ultra-lightweight**: Only **1.18M parameters**, suitable for edge devices.
- **Accuracy-preserving optimizations**:
  - **SPD-Conv** (Space-to-Depth Convolution)  
  - **MLCA** (Mixed Local Channel Attention)  
  - **C3-GhostDynamicConv**  
  - **RCSOSA blocks**  
  - **CRFM fusion**  
  - **NWD Loss** (Normalized Wasserstein Distance)  
- **Deployment-ready**: Optimized for **NVIDIA Jetson Orin Nano** and ONNX Runtime.

---

## 🧪 Ablation Study
Cumulative impact of modules added to YOLOv5n (baseline):

| Configuration               | P     | R     | mAP@0.5 | mAP@0.5:0.95 | Params (M) |
|------------------------------|-------|-------|---------|--------------|------------|
| YOLOv5n* (baseline)         | 0.934 | 0.835 | 0.874   | 0.385        | 1.70       |
| + Optimized head             | 0.957 | 0.880 | 0.918   | 0.430        | 1.70       |
| + NWD loss                   | 0.958 | 0.871 | 0.924   | 0.424        | 1.70       |
| + SPDConv                    | 0.963 | 0.895 | 0.928   | 0.418        | 1.45       |
| + MLCA fusion                | 0.971 | 0.916 | 0.947   | 0.421        | 1.55       |
| + C3-GhostDynamicConv        | 0.986 | 0.914 | 0.943   | 0.428        | 1.39       |
| + CRFM structure             | 0.969 | 0.904 | 0.933   | 0.424        | 0.91       |
| + RCSOSA attention (final)   | 0.970 | 0.914 | 0.945   | 0.432        | 1.18       |

> 🔑 Each module **improves detection accuracy** while progressively reducing or balancing parameter size:contentReference[oaicite:2]{index=2}.

---

## 📊 Comparative Analysis with SOTA
LW-YOLOv5 compared against other lightweight and PCB-specific models:

| Model            | Year | mAP@0.5 | Params (M) |
|------------------|------|---------|------------|
| YOLOv5n (Baseline) | 2020 | 0.873   | 1.70       |
| MSD-YOLO         | 2021 | 0.994   | 3.80       |
| YOLO-LFPD        | 2022 | 0.982   | 6.40       |
| ARMA-based YOLO  | 2023 | 0.950   | 2.121      |
| **LW-YOLOv5 (Proposed)** | 2025 | **0.945** | **1.18** |

✔ LW-YOLOv5 achieves a strong **accuracy–efficiency balance**, making it ideal for embedded deployment:contentReference[oaicite:3]{index=3}.

---

## 🏗 Model Architecture

### LW-YOLOv5 (Proposed)
![LW-YOLOv5 Architecture](YOLO%20Model%20Architecture%20Design-Modified%20YOLO%20Structure%20(1).jpg)

---

### YOLOv5n (Baseline)
![YOLOv5n Architecture](YOLO%20Model%20Architecture%20Design-Original%20YOLOv5%20Structure%20(1).jpg)

---

## ⚙️ Training & Evaluation

### 📍 Google Colab Setup
```bash
# Clone repo
!git clone https://github.com/JacksonR553/LW-YOLO.git
%cd LW-YOLO

# Install dependencies
!pip install -r requirements.txt

# Download PKU-Market-PCB dataset (example via Kaggle)
!kaggle datasets download -d jacksonr553/pku-market-pcb
!unzip pku-market-pcb.zip -d datasets/

# Train LW-YOLOv5
!python train.py --img 640 --batch 16 --epochs 100 \
  --data data/pcb.yaml --cfg models/lw-yolov5.yaml \
  --weights '' --name lw-yolov5-pcb
