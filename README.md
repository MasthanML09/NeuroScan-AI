# ***NeuroScan AI — Brain Tumor Detection*** [![Live App](https://img.shields.io/badge/Open_App-black?style=for-the-badge&logo=streamlit)](https://neuroscan-ai-gvdia.streamlit.app/)
**A deep learning system for automated brain tumor detection from MRI scans, built with VGG16 transfer learning and deployed as an interactive web application.**
---

## Overview

NeuroScan AI classifies brain MRI scans into four categories — glioma, meningioma, pituitary tumor, and no tumor — using a fine-tuned VGG16 convolutional neural network. The system achieves 95%+ test accuracy and provides Grad-CAM visual explanations showing exactly which regions of the scan influenced the prediction.

---

## Demo

Upload any brain MRI image and the app returns the tumor type, confidence score, probability distribution across all classes, and a heatmap overlay highlighting the regions the model attended to.

---

## Model Architecture

The model uses VGG16 pre-trained on ImageNet as a frozen feature extractor, with a custom classification head trained on the Brain Tumor MRI Dataset.
```
Input (128 x 128 x 3)
    VGG16 backbone (frozen)
    Flatten
    Dropout(0.4) → Dense(256, ReLU)
    Dropout(0.3) → Dense(128, ReLU)
    Dropout(0.2) → Dense(4, Softmax)
```
Training followed a two-phase strategy. Phase one trained only the classification head with the base frozen at a learning rate of 1e-3. Phase two unfroze VGG16's final convolutional block and fine-tuned the entire network at 1e-5. Both phases used early stopping, learning rate reduction on plateau, and model checkpointing.

---

## Dataset

Brain Tumor MRI Dataset by Masoud Nickparvar (Kaggle)
7,000+ MRI images across four classes with pre-defined train/test splits.

Classes: `glioma` · `meningioma` · `pituitary` · `notumor`

---

## Results

| Metric | Score |
|---|---|
| Test Accuracy | 95%+ |
| Macro F1-Score | 0.95 |
| AUC-ROC (avg) | 0.97+ |

Meningioma is the most challenging class due to visual similarity with surrounding tissue — per-class metrics are reported in the notebook.

---

## Project Structure
```
NeuroScan-AI/
├── NeuroScan_AI.py           — Streamlit web application
├── Brain_Tumor_prediction.ipynb  — Training pipeline (Kaggle)
├── brain_tumor_model.keras   — Trained model weights
├── label_map.json            — Class index to label mapping
└── requirements.txt          — Python dependencies
```
---

## Local Setup
```bash
git clone https://github.com/masthanshaik2201-tech/NeuroScan-AI.git
cd NeuroScan-AI

python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
streamlit run NeuroScan_AI.py
```

---

## Key Technical Decisions

**Why Functional API over Sequential** — Keras 3.x has a known bug where wrapping a functional model like VGG16 inside a Sequential container causes Flatten to receive a list instead of a tensor. The entire pipeline uses the Functional API to avoid this.

**Why load_weights instead of load_model** — Decouples architecture definition from saved weights, making the app robust to Keras version differences across environments.

**Label encoding** — Class-to-index mapping is saved as `label_map.json` at training time using `sorted(os.listdir())` to guarantee deterministic ordering. This prevents the silent bug where predictions map to wrong class names across different runs.

**Grad-CAM** — Implemented using the `block5_conv3` layer of VGG16, accessed via `model.get_layer('vgg16')` rather than layer index to ensure compatibility regardless of model wrapping.

---


## Tech Stack

`Python` · `TensorFlow` · `Keras` · `VGG16` · `scikit-learn` · `Streamlit` · `NumPy` · `Matplotlib` · `Pillow`

---

## Disclaimer

This tool is built for educational and research purposes only. It is not a substitute for professional medical diagnosis. Always consult a qualified neurologist or radiologist for clinical decisions.
