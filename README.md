# 🎭 Emotion Detection from Speech using Deep Learning

This project focuses on detecting human emotions from speech signals using an LSTM-based deep learning model. It leverages audio signal processing and machine learning techniques to classify emotional states such as **Happy**, **Sad**, **Angry**, **Fearful**, and **Neutral**.

---

## 🚀 Features

- 🎤 **Real-Time Speech Emotion Recognition**  
- 🧠 **LSTM-based Deep Learning Model**
- 📊 **Waveform Visualization and Emotion Distribution Charts**
- ⚡ **Streamlit-based UI for Instant Interaction**
- 🔍 **MFCC Feature Extraction using Librosa**
- 📁 **Supports both real-time microphone input and pre-recorded files**

---

## 📊 Model Overview

- **Architecture**: LSTM
- **Input**: MFCC Features (20 coefficients)
- **Dataset**: RAVDESS + Custom Augmented Dataset
- **Accuracy**: High accuracy with well-separated emotion classes

---

## 🛠️ Installation

git clone https://github.com/Prathmesh0001761/Emotion-detections-updated.git

cd Emotion-detections-updated

pip install -r requirements.txt

▶️ Run the App

streamlit run app.py

📌 Dependencies

Python 3.8+

TensorFlow 2.x

Streamlit

Librosa

NumPy

Soundfile

Matplotlib

Install all dependencies using:

pip install -r requirements.txt

💡 Future Enhancements

🎯 Improve accuracy with attention mechanism

🎙️ Add support for longer conversations

🧪 Integrate with chatbot for emotion-aware responses
