
# 🎓 EPFL Project – Eco-Friendly Hate Speech Detection in Video & Audio

## 🔎 Overview

This project provides an **intelligent and environmentally conscious platform** for detecting hate speech in **videos** and **audio** using a simple **Gradio/hugging Face interface**.

It combines the latest tools in **NLP**, **emotion analysis** and **computer vision**, with **CO₂ tracking**, to offer both performance and eco-responsibility.

---

## 👥 Team

Developed at EPFL as part of an academic project.

**Authors:**
- Loris Alan Fabbro  
- Mohammed Al-Hussini  
- Loic Misenta  

---
## Dataset

We use the HateMM dataset, consisting of ~43 hours of manually annotated videos from BitChute, labeled as hate or non-hate with relevant frame spans.
This dataset was introduced in the paper "HateMM: A Multi-modal Dataset for Hate Video Classification."
We thank the authors for making this resource publicly available.
---

## 🚀 Key Features

✅ Upload **videos (.mp4)** or **audios (.wav/.mp3)**  
✅ 3-layer detection system (Audio / Images samples / OCR-Text)  
✅ Interactive results: **clickable hate segments**  
✅ Tracks and displays **carbon footprint** of analysis  
✅ Customizable eco modes:
- `Low 🌱` – Minimal emissions  
- `Medium ♨️` – Balanced  
- `High ⚠️` – Maximum analysis  

---

## 🧠 System Architecture

| Stage              | Description                                                              |
|-------------------|--------------------------------------------------------------------------|
| **1. Audio Filter**   | `Whisper` (transcription) + `SenseVoiceSmall` (emotion) + `EmoHateBERT` (hate speech detection) |
| **2. Visual Filter**  | `CLIP` to detect hateful gestures, scenes, or signs etc                  |
| **3. OCR Filter**     | `EasyOCR` + `hatebert` to catch hate in text from video frames       |
| **Fusion Layer**      | Merges and expands detected segments across filters                   |
| **CO₂ Tracker**       | Monitors emissions via `codecarbon` or `EmissionsTracker`             |

---

## 🎛️ Interface (Gradio/Hugging_face)
link : https://huggingface.co/spaces/Lorissss/Detection_hate_speech
The interface lets you:
- Upload **a video or audio file**
- Choose the **eco mode** and run the **hate speech detection**
- See **clickable hate timestamps**
- View a summary:
  - total duration
  - number of segments
  - CO₂ emissions

Example output:
```
[VIDEO]
[AUDIO]
🕑 Segment 1: 00:01:15 ➔ 00:01:47
🕑 Segment 2: 00:05:05 ➔ 00:05:21
⏳ Total Hate Speech Duration: 0:05:30
♻️ Carbon Footprint: 0.034g

```

---

## 📁 Project Structure

```

.
├── app.py                  # Gradio interface
├── Implementation.py       # Full processing logic (audio, vision, OCR, NLP)
├── model.py                # Pretrained model (.pt) + `student_distilled_EmoHateBERT.pt` + `EmoHateBert_teacher.pt`
├── ctc_alignement.py 
├── pipeline.png            # Optional pipeline diagram
├── loading.gif             # Custom loading animation
├── requirements.txt        # Python dependencies
└── README.md               # This file

````

---

## 🛠️ How to Run

1. **Clone the repository**:

```bash
git clone <Our repository>

````

2. **Install the dependencies**:

```bash
pip install -r requirements.txt
```

3. **Place the required model files** into the `repository` folder:
(link : https://www.swisstransfer.com/d/755fdc17-c422-477c-be3f-62369a37105c or send email: loris.fabbro@epfl.ch)
   * `student_distilled_EmoHateBERT.pt`
   * `EmoHateBert_teacher.pt`

5. **Launch the app**:

```bash
python app.py
```

Then open the Gradio link provided (usually `http://localhost:7860/`).

---

## 📦 Tech Stack & Models Used

* 🤖 `Whisper` – Audio transcription
* 🎭 `SenseVoiceSmall` – Emotion recognition
* 🧠 `EmoHateBERT` – Hate classification with emotion-aware BERT
* 🖼️ `CLIP` – Vision-text similarity detection
* 🔤 `EasyOCR` – Text recognition in frames
* 🧾 `hatebert` – Hate detection in OCRed text
* 🌍 `codecarbon` – Carbon footprint estimation

---

## ⚠️ Disclaimer

* For **academic and research purposes only**.
* Not certified for legal moderation or production-level deployment.
* May inherit biases from underlying models.

---

## 📜 License

This project was developed as part of an **EPFL academic assignment**.
MIT License

---

