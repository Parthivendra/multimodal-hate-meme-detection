# 🧠 Multimodal Hate Meme Detection

A deep learning project for detecting offensive/hateful memes using **multimodal learning (text + image)**. This system combines natural language processing and computer vision to better understand meme context.

---

## 🚀 Overview

Memes convey meaning through a combination of **text and visual context**. Traditional models often fail to capture this interaction.

This project builds a **multimodal pipeline** that integrates:

* Text understanding using DistilBERT
* Image understanding using ResNet-18
* Feature fusion for joint classification

---

## 🎯 Objective

To classify memes into four categories:

* `not_offensive`
* `slight`
* `very_offensive`
* `hateful_offensive`

---

## 📂 Dataset

* **Memotion Dataset**
* Contains:

  * Meme images
  * OCR-extracted and corrected text (`text_corrected`)
  * Labels for offensiveness

---

## 🧠 Model Architecture

### 🔹 Text Model

* DistilBERT (pretrained)
* CLS token representation
* Fully connected classifier

### 🔹 Image Model

* ResNet-18 (pretrained on ImageNet)
* Backbone frozen
* Modified final layer

### 🔹 Multimodal Model

* Text features: 768-dim
* Image features: 512-dim
* Concatenation → 1280-dim
* Fully connected fusion network with dropout

---

## 🏗️ System Pipeline

```
Text + Image → Preprocessing → Encoders (BERT + ResNet) → Feature Fusion → Classifier → Prediction
```

---

## ⚙️ Tech Stack

* PyTorch
* HuggingFace Transformers
* Torchvision
* Scikit-learn
* Kaggle (GPU training)
* GitHub (version control)

---

## 🔄 Workflow

```
Local Development ⇄ GitHub ⇄ Kaggle
```

* Code written locally
* Synced via GitHub
* Training performed on Kaggle GPU
* Results pulled back locally

---

## 🧪 Training Details

* Loss: CrossEntropyLoss (with class weights)
* Optimizer: AdamW
* Learning Rates:

  * Text: `2e-5`
  * Image: `1e-5`
  * Multimodal: `2e-5`
* Metrics:

  * Accuracy
  * Weighted F1 Score

---

## 📊 Results Summary

| Model      | Performance                                              |
| ---------- | -------------------------------------------------------- |
| Text-only  | Moderate performance (limited by lack of visual context) |
| Image-only | Limited performance (semantic gap in meme understanding) |
| Multimodal | Improved performance due to combined features            |

> Multimodal learning significantly outperforms individual modalities.

---

## ⚠️ Challenges Faced

* Missing/corrupt images in dataset
* Class imbalance
* Weak performance of single-modality models
* GPU compatibility issues on Kaggle

---

## 🔮 Future Improvements

* Use CLIP-based multimodal embeddings
* Add attention-based fusion
* Improve data balancing
* Deploy as a web app (Streamlit/Flask)
* Fine-tune entire image backbone

---

## 📦 Project Structure

```
multimodal-hate-meme-detection/
│
├── src/
│   ├── dataset.py
│   ├── model_text.py
│   ├── model_image.py
│   ├── model_multimodal.py
│   ├── train.py
│
├── notebooks/
├── data/
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

### 1. Clone Repository

```bash
git clone https://github.com/your-username/multimodal-hate-meme-detection.git
cd multimodal-hate-meme-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train Model

* Upload repository to Kaggle
* Add Memotion dataset
* Run training notebook

---

## 🔍 Inference Example

```python
pred = predict_meme(text, image_path, model, tokenizer, transform, device)
print(pred)
```

---

## 📚 References

* Memotion Dataset
* DistilBERT (HuggingFace)
* ResNet (He et al., 2015)
* Multimodal Learning Research

---

## 👤 Author

**Parthivendra Singh**

---

## ⭐ Final Note

This project demonstrates the importance of **multimodal learning** in understanding complex real-world data like memes, where meaning emerges from the interaction between text and visuals.
