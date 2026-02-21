# 🎧 PyTorch Audio Processing

A basic starter project for performing **audio processing and classification using PyTorch**.

This repository demonstrates:
- Loading audio files
- Converting audio into Mel Spectrograms
- Creating a custom Dataset
- Training a simple CNN model
- Running inference

---

## 📦 Requirements

- Python 3.8+
- torch
- torchaudio
- librosa (optional)
- numpy
- matplotlib

Install dependencies:

```bash
pip install torch torchaudio librosa numpy matplotlib


📁 Project Structure

project/
│
├── data/
│   ├── train/
│   └── test/
│
├── dataset.py
├── model.py
├── train.py
├── inference.py
└── README.md


🔊 1. Loading Audio

import torchaudio

waveform, sample_rate = torchaudio.load("audio.wav")
print("Shape:", waveform.shape)
print("Sample Rate:", sample_rate)

🎼 2. Convert to Mel Spectrogram

import torchaudio.transforms as T

mel_transform = T.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=512,
    n_mels=128
)

mel_spec = mel_transform(waveform)
print(mel_spec.shape)

🗂️ 3. Custom Dataset

from torch.utils.data import Dataset
import torchaudio
import os

class AudioDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.file_paths[idx])
        return waveform, self.labels[idx]


```

Common Audio Features

Mel Spectrogram

Log-Mel Spectrogram

MFCC

Chroma

Spectral Contrast

🎯 Applications

Music genre classification

Speech emotion recognition

Environmental sound classification

Instrument detection

Noise-robust audio classification

⚙️ Best Practices

Normalize audio

Use fixed-duration clips (5–10 sec)

Apply augmentation (noise, time shift, pitch shift)

Convert spectrograms to log scale

Use pretrained models for better performance

📌 Future Improvements

Add data augmentation

Add pretrained models (AST / HTS-AT / BEATs)

Use mixed precision training

Implement early stopping

Add model checkpoint saving
