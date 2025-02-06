# 🖼️ Image Captioning using CNN + LSTM  
### 🚀 TechA - Machine Learning Internship (Level 02, Task 01)  

![Image Captioning](https://upload.wikimedia.org/wikipedia/commons/3/3e/Deep_learning_captioning_example.jpg)  

## 📌 Overview  
This project focuses on **Image Captioning**, where a **deep learning model** generates textual descriptions for images. The project integrates **Computer Vision (CNN - InceptionV3)** with **Natural Language Processing (LSTM-based RNN)** for caption generation.  

- ✅ **Feature Extraction:** CNN (**InceptionV3**)  
- ✅ **Text Generation:** LSTM-based RNN  
- ✅ **Word Embeddings:** **GloVe Pre-trained Word Embeddings**  
- ✅ **Dataset:** Images + Captions  

---

## 📂 Dataset  
- **Images stored in:** `/content/dataset/Images/`  
- **Captions stored in:** `/content/dataset/captions.txt`  

---

## 🛠️ Technologies Used  
✅ **Python**  
✅ **TensorFlow / Keras**  
✅ **InceptionV3 (CNN for Feature Extraction)**  
✅ **LSTM (RNN for Caption Generation)**  
✅ **GloVe Pre-trained Word Embeddings**  
✅ **Google Colab (Free Version)**  

---

## 📌 Steps to Run the Project  

### 2️⃣ Load Dataset from Google Drive  
- **Mount Google Drive**  
- **Unzip Dataset Files**  

```python
import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

from google.colab import drive
drive.mount('/content/drive')

# Unzip dataset
!unzip -q '/content/drive/MyDrive/archive (7).zip' -d '/content/dataset'

# Check dataset structure
!ls /content/dataset
