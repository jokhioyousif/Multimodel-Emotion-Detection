# 🎭 Multimodal Emotion Detection App

This project is an **AI-powered application** that can detect human emotions across multiple input types — **text, face (image/video), and voice**. It combines state-of-the-art models from Hugging Face with an interactive **Gradio** interface, deployed on **Hugging Face Spaces**.

---

## 🚀 Features

* **Text Emotion Analysis** → Detects emotions like joy, sadness, anger, fear, etc. from written text.
* **Face Emotion Recognition** → Identifies facial expressions from webcam input or uploaded images.
* **Voice Emotion Analysis** → Recognizes emotions from short audio clips or microphone recordings.
* **Video Emotion Detection** → Samples frames from uploaded or recorded videos and averages detected emotions.
* **Interactive UI** → Built with Gradio Tabs for a clean and easy-to-use design.

---

## 🛠️ Tech Stack

* **Python**
* **Hugging Face Transformers**
* **Gradio** (UI framework)
* **PyTorch**
* **PIL, ImageIO** (for image/video handling)
* **Torchaudio, Soundfile** (for audio processing)

---

## 📂 Project Structure

```
.
├── app.py                  # Main multimodal app (Text, Face, Voice, Video)
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

---

## 📦 Installation & Run Locally

1. Clone the repo:

   ```bash
   git clone https://huggingface.co/spaces/your-username/multimodal-emotion-detection
   cd multimodal-emotion-detection
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   python app.py
   ```

---

## 🌐 Deployment on Hugging Face Spaces

This app is ready to be deployed on **Spaces**:

1. Create a new Space on Hugging Face.
2. Upload `app.py`, `app_text_only_backup.py`, `requirements.txt`, and `README.md`.
3. Select **Gradio** as the SDK.
4. Run the Space — your app will be live! 🚀

---

## 🎯 Example Models Used

* **Text** → [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)
* **Image/Video** → [`trpakov/vit-face-expression`](https://huggingface.co/trpakov/vit-face-expression)
* **Audio** → [`superb/hubert-large-superb-er`](https://huggingface.co/superb/hubert-large-superb-er)

---

## ✨ Future Improvements

* Support for **multilingual text emotion detection**.
* Better **real-time video analysis**.
* Combining all modalities for **fusion-based emotion prediction**.

---

## 👨‍💻 Author

Developed by **Muhammad Yousif** — AI Engineer passionate about multimodal deep learning and interactive AI apps.
