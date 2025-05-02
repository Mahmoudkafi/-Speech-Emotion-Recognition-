# Speech Emotion Recognition using CNN + BiLSTM (TESS Dataset)

This project implements a **Speech Emotion Recognition (SER)** system that classifies human emotions from voice recordings using a deep learning model built with **CNN and BiLSTM** layers.

The model is trained on the **TESS (Toronto Emotional Speech Set)** dataset and utilizes advanced audio preprocessing and data augmentation techniques for robust classification.

---

## 📁 Dataset

- **Source**: [TESS Dataset](https://dataverse.library.yorku.ca/dataset.xhtml?persistentId=doi:10.5683/SP2/E8H2MF)
- **Classes**: Angry, Disgust, Fear, Happy, Neutral, Pleasant Surprise, Sad
- **Format**: `.wav` files labeled by emotion in the filename.

---

## 📊 Features Extracted

- **MFCC (Mel Frequency Cepstral Coefficients)**
- Audio length padded and fixed to a uniform shape
- Sampling rate: `22050 Hz`

---

## 🎛️ Data Augmentation Techniques

To increase generalization and avoid overfitting:
- `Noise Injection`
- `Pitch Shifting`
- `Time Stretching`

---

## 🧠 Model Architecture

- `Conv1D + MaxPooling + BatchNormalization`
- `Bidirectional LSTM` layers
- `Dense` output with `Softmax` activation

Compiled using:
```python
loss = 'categorical_crossentropy'
optimizer = 'adam'
metrics = ['accuracy']
```

---

## 📈 Training Configuration

- **Epochs**: 50  
- **Batch Size**: 32  
- **Validation Split**: 20%  
- **Early Stopping**, **ReduceLROnPlateau**, and **ModelCheckpoint** used

---

## 📂 Project Structure

```
numbered_ser_project.py        # Main script
best_model.keras               # Trained model file
preprocessed_data.pkl          # Optional pickled features
README.md                      # Project documentation
```

---

## 🧪 Predicting New Samples

You can predict emotion from a folder of `.wav` files:

```python
predict_emotion_from_folder("path/to/emotion_folder", num_samples=10)
```

---
