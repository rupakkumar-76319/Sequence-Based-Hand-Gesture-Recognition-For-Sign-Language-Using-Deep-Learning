# Sequence-Based Hand Gesture Recognition for Sign Language using Deep Learning

> **B.Tech Project — Phase 1**  
> Department of Electronics and Communication Engineering  
> Triguna Sen School of Technology, Assam University, Silchar  

---

## 👥 Authors
   $ Rupak Kumar 



## 📌 About the Project

This project presents a **vision-based, markerless system** for American Sign Language (ASL) gesture recognition using **MediaPipe Hands** for real-time 3D hand keypoint extraction. Every gesture — whether static or dynamic — is represented as a **temporal sequence of 30 frames**, each containing **126 hand landmark features**.

Six deep learning architectures are systematically implemented, trained, and evaluated under identical experimental conditions on a 36-class ASL dataset (digits 0–9 and alphabets A–Z).

This is **Phase 1** of a broader research initiative. The next phase targets fully dynamic ASL word/phrase recognition and Text-to-Sign conversion.

---

## 🎯 Objectives

1. Extract 3D hand keypoints from webcam video using **MediaPipe Hands** and build a labelled dataset of 36 ASL gesture classes
2. Design and train **six deep learning architectures** for sequence classification
3. Evaluate models using accuracy, precision, recall, F1-score, FPR, and inference time
4. Deploy the best-performing model in a **real-time inference pipeline** using OpenCV

---

## 🧠 Models Implemented

| Model | Type |
|-------|------|
| LSTM | Recurrent |
| GRU | Recurrent |
| BiLSTM | Recurrent (Bidirectional) |
| 1-D CNN | Convolutional |
| CNN-GRU | Hybrid |
| CNN-BiLSTM | Hybrid |

---

## 📊 Results

| Model | Accuracy | F1 Score | Recall | Loss | Time (s) | Rank |
|-------|----------|----------|--------|------|----------|------|
| **CNN** | **92.68%** | **0.9248** | **0.9268** | **0.2205** | 0.0218 | 🥇 1 |
| CNN-BiLSTM | 92.17% | 0.9206 | 0.9217 | 0.2354 | **0.1175** | 🥈 2 |
| CNN-GRU | 86.62% | 0.8615 | 0.8662 | 0.4514 | 0.1441 | 🥉 3 |
| BiLSTM | 85.86% | 0.8431 | 0.8586 | 0.3631 | 0.5850 | 4 |
| LSTM | 83.33% | 0.8159 | 0.8333 | 0.4731 | 0.4165 | 5 |
| GRU | 83.33% | 0.8255 | 0.8333 | 0.4808 | 0.8367 | 6 |

> ✅ **CNN** — Best overall accuracy (92.68%)  
> ⚡ **CNN-BiLSTM** — Recommended for real-time deployment. It work well on the large dataset.
> 🎯 **Top-3 Accuracy** — ~98%

---

## 📁 Project Structure

```
Phase1/
  app/
    backend/
      sign_to_text_realtime.py   # real-time inference
    frontend/
      index.html
  configs/
    training_config.yaml
  data/
    dataset.csv                  # processed keypoint dataset
    Convert/
      numpy_to_csv.py
    objects/
      label_encoder.pkl
    processed/                   # per-class .npy files
  dataloader/
    dataset.py
    dataloader.py
  experiments/
    results/                     # classification reports, plots
  models/
    Models/
      model_LSTM.py
      model_GRU.py
      model_BiLSTM.py
      model_CNN.py
      model_CNNGRU.py
      model_CNNBiLSTM.py
      evaluate.py
      compare_Models.py
  preprocessing/
    hand_detection.py
    normalize.py
  create_encoder.py
```

---

## 🗃️ Dataset

| Property | Value |
|----------|-------|
| Total Samples | 2634 |
| Classes | 36 (A–Z and 0–9) |
| Frames per Sequence | 30 |
| Features per Frame | 126 (21 landmarks × 3 coordinates × 2 hands) |
| Feature Shape | (30, 126) |
| Train / Val / Test | 70% / 15% / 15% |
| Train Samples | 1843 |
| Val Samples | 395 |
| Test Samples | 396 |
| Augmentation | Mirror flipping (horizontal flip of x-coordinates) |
| Storage Format | CSV (3781 columns) |

- **Collection:** Webcam + MediaPipe Hands pipeline
- **Reference:** American Sign Language (ASL)
- **Class Imbalance:** Min 41 samples (class E) — Max 88 samples (class Z)


Dataset is hosted on Kaggle due to GitHub file-size limitations.

Kaggle Dataset:
https://www.kaggle.com/datasets/rupakkumar24f2001886/sequence-based-hand-for-sign-language

---

## ⚙️ Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.10.20 | Programming language |
| PyTorch | 2.1.2 | Model training & inference |
| MediaPipe | 0.10.14 | Hand keypoint extraction |
| OpenCV | 4.9.0.80 | Video capture & display |
| Scikit-learn | 1.4.2 | Label encoding, metrics |
| NumPy | 1.26.4 | Numerical operations |
| Pandas | 2.1.4 | Data loading |
| Matplotlib | 3.10.8 | Evaluation plots |
| Seaborn | 0.13.2 | Advanced evaluation plots |

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install torch mediapipe opencv-python scikit-learn numpy pandas matplotlib seaborn
```

### 2. Prepare Dataset
```bash
python create_encoder.py
python data/Convert/numpy_to_csv.py
```

### 3. Train a Model
```bash
# Example: train CNN model
python models/Models/model_CNN.py
```

### 4. Compare All Models
```bash
python -m models.compare_Models
```

### 5. Real-Time Inference
```bash
python app/backend/sign_to_text_realtime.py
```

---

## 🏗️ System Pipeline

```
Camera → MediaPipe Hands → Normalization → Feature Vector
       → 30-Frame Buffer → Sequence Input → Model
       → Post Processing → Output Label (Real-Time Text)
```

- Predictions smoothed by **majority voting** over 15 consecutive predictions
- Only predictions with **confidence > 0.6** are accepted

---

## 🔮 Future Work

1. Extend to fully **dynamic ASL words and phrases**
2. Implement **Text to Sign** conversion (two-way communication)
3. Explore **Transformer and Attention-based** architectures
4. Apply data augmentation to fix **class imbalance**
5. Incorporate **facial expressions and body posture**
6. Deploy on **mobile/embedded devices** using TensorFlow Lite or ONNX
7. Extend to **Indian Sign Language (ISL)**

---
## 📄 License

This project is licensed under the **MIT License** — see the 
[LICENSE](LICENSE) file for details.

Copyright (c) 2025 Rupak Kumar

---
