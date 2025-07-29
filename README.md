# 🦎 Camouflaged Object Detection using Deep Learning

## 📑 Technical Seminar Presentations
- Phase I: ![Technical Seminar Phase I](reports/Technical_Seminar_Phase_I.pdf)
- Phase II: ![Technical Seminar Phase II](reports/Technical_Seminar_Phase_II.pdf)

---

## 🎯 Project Overview
This project implements **Camouflaged Object Detection (COD)** using a **U-Net segmentation model** trained on the CAMO dataset.

### Key Features:
- Image preprocessing & augmentation
- Training U-Net for segmentation
- Evaluation using IoU (Jaccard index)
- Visual prediction overlays

---

## 📂 Dataset
[CAMO Dataset](https://www.kaggle.com/datasets/ivanomelchenkoim11/camo-dataset).

---

## 🚀 Getting Started
1️⃣ Clone repository:
```bash
git clone https://github.com/your-username/camouflaged-object-detection.git
cd camouflaged-object-detection
```

2️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```

3️⃣ Download dataset:
```bash
python data/download_dataset.py
```

4️⃣ Train the U-Net model:
```bash
python src/train_unet.py
```

5️⃣ Evaluate model:
```bash
python src/evaluate.py
```

---

## 🛠 Tech Stack
- TensorFlow/Keras
- OpenCV
- NumPy, Matplotlib
- Scikit-learn

---

## 📜 License
MIT License © 2025
