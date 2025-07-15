# BRAIN TUMOR DETECTOR USING HYBRID QUANTUM+ML MODEL

- This project presents a cutting-edge medical diagnostics system that uses a **Hybrid Quantum Neural Network (QNN)** combined with **classical MLP layers** to detect brain tumors from medical images. It leverages quantum computing principles through **PennyLane** and **PyTorch**, making it a unique fusion of classical AI and quantum machine learning (QML).
---
## Abstract

- Brain tumor detection is a critical challenge in medical diagnostics. This project explores a **quantum-enhanced approach** to improve prediction accuracy using **hybrid QNN models**. The system was trained and tested on a labeled dataset of brain MRI images and achieves a decent classification accuracy while experimenting with various qubit configurations and encoding techniques.
---
## Features:

- Predicts tumor presence from MRI image data
- Hybrid QNN + MLP neural architecture using PennyLane and PyTorch
- Streamlit app for easy user interface and testing
---
## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/V-Preetha/Brain-Tumor-Detector-Quantum-ML-.git
cd Brain-Tumor-Detector-Quantum-ML-
```
### 2. Kaggle Dataset used
```
https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset/data
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```
### 4. Run the .py files in this particular order:
```bash
python main.py
python qnn_model.py
```

### 5. Train the model
```bash
python train_qnn.py
```

### 6. Launch the StreamLit app:
```bash
python -m streamlit run app.py
```

## DEMO


https://github.com/user-attachments/assets/fc58d2f9-3e89-4932-aa5b-8ae856551759

