# LW-YOLOv5: Lightweight CNN for PCB Defect Detection

![LW-YOLOv5 Architecture](https://github.com/user-attachments/assets/352d1650-e72c-49ed-bca6-2af44ef1bb15)

---

## 📌 Introduction
Printed Circuit Board (PCB) defect detection is crucial for ensuring quality in electronic manufacturing. Traditional inspection methods (manual or AOI) are error-prone and inefficient. Deep learning–based object detection has emerged as a reliable alternative, but deploying these large models on **embedded devices** (such as NVIDIA Jetson Orin Nano) is challenging due to **limited computing power and memory**.

This project proposes **LW-YOLOv5**, an **optimized lightweight version of YOLOv5n**, designed for **real-time PCB defect detection on embedded devices**.  

Key highlights:
- Reduced parameters from **1.70M → 1.18M** (–31%).
- Achieved **mAP@0.5 = 0.945** with only 1.18M parameters.
- Outperforms many SOTA models in **accuracy–efficiency trade-off**.

## 📂 Dataset
- PKU-Market-PCB dataset used for training & evaluation.
- Six PCB defect classes: missing holes, mouse bites, open circuits, shorts, spurs, spurious copper
- Dataset can be found on Kaggle(https://www.kaggle.com/datasets/akhatova/pcb-defects)

## 🔬 Key Contributions
- Developed LW-YOLOv5: A compact model (1.18M params) optimized for embedded devices.
- Conducted comprehensive ablation studies proving the cumulative benefits of each lightweight module.
- Performed comparative studies against state-of-the-art models.
- Achieved robust deployment on embedded hardware with real-time inference and low power usage.

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

> 🔑 Each module **improves detection accuracy** while progressively reducing or balancing parameter size.

## 📊 Comparative Analysis with SOTA
LW-YOLOv5 compared against other lightweight and PCB-specific models:

| Model            | Year | mAP@0.5 | Params (M) |
|------------------|------|---------|------------|
| YOLOv5n (Baseline) | 2020 | 0.873   | 1.70       |
| MSD-YOLO         | 2021 | 0.994   | 3.80       |
| YOLO-LFPD        | 2022 | 0.982   | 6.40       |
| ARMA-based YOLO  | 2023 | 0.950   | 2.121      |
| **LW-YOLOv5 (Proposed)** | 2025 | **0.945** | **1.18** |

✔ LW-YOLOv5 achieves a strong **accuracy–efficiency balance**, making it ideal for embedded deployment.

## 🏗 Model Architecture

### LW-YOLOv5 (Proposed)
![LW-YOLOv5 Architecture](https://github.com/user-attachments/assets/352d1650-e72c-49ed-bca6-2af44ef1bb15)

### YOLOv5n (Baseline)
![YOLOv5n Architecture](https://github.com/user-attachments/assets/1b33e95a-70d7-4056-a55e-079c2627405c)

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

# Evaluate model performance
!python val.py --weights runs/train/lw-yolov5-pcb/weights/best.pt \
  --data data/pcb.yaml --img 640
```

## 📦 Deployment on Jetson Orin Nano
```bash
# Export to ONNX
!python export.py --weights runs/train/lw-yolov5-pcb/weights/best.pt \
  --include onnx --dynamic --simplify

# Run inference (PyTorch)
!python detect.py --weights best.pt --img 640 \
  --conf 0.25 --source pcb_yolo_dataset/images/test/

# Run inference (ONNX - FP16)
!python detect.py --weights best.onnx --img 640 \
  --conf 0.25 --half --source pcb_yolo_dataset/images/test/
```

## 📈 Experimental Results & Visualization

To validate LW-YOLOv5, we performed extensive evaluation on the **PKU-Market-PCB** dataset. Below are some key visualizations from the experiments:

### 🔹 Precision–Recall Curve
<img width="755" height="660" alt="confusion matrix" src="https://github.com/user-attachments/assets/8110335f-069a-4538-8750-f91dd4e164ca" />

### 🔹 Confusion Matrix
<img width="828" height="481" alt="precision and recall chart" src="https://github.com/user-attachments/assets/8f604454-da14-4434-9126-6e1f660e5182" />

### 📊 Key Metrics
- **mAP@0.5**: **0.945**  
- **Precision**: **0.970**  
- **Recall**: **0.914**  
- **Parameters**: **1.18M**  
- **GFLOPs**: **5.1**  

These results show LW-YOLOv5 achieves **high accuracy while remaining extremely lightweight**, making it ideal for **embedded deployment** on devices like the NVIDIA Jetson Orin Nano.
