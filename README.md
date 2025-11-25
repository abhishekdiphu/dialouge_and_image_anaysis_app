Below is your **README.md written fully in proper Markdown syntax**, ready to save as a file named **`README.md`**.

---

# 📘 Multimodal AI Web Application (Flask)

This project is a **Flask-based multimodal AI system** capable of performing:

* 🖼️ **Image Explanation / Captioning**
* 💬 **Dialogue Summarization**

It integrates multiple transformer models and custom logic for handling text, vision, and conversational interactions.

---

## 🚀 Features


### ✔ Image Explanation

Uploads an image with a custom prompt to generate a detailed description using a Vision-Language transformer model.

### ✔ Dialogue Summarization

Collects and formats conversation turns between *Human* and *AI* and produces a summary.

---

## 📦 Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd <your-project-folder>
```

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate       # macOS / Linux
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔧 Project Structure

```
project/
│
├── sentiment_analysis/
│   └── sentiment_analysis.py
├── image_captioning/
│   └── image_analysis.py
├── text_summerization/
│   └── summerizaton.py
│├── static/
│   └── main_page.js
├── templates/
│   └── index.html
│
├── server.py
├── requirements.txt
└── README.md
```

---

## ▶️ Running the Application

Launch the Flask server:

```bash
python3 server.py
```

By default, the server runs on:

```
http://0.0.0.0:9000
```

Open in your browser:

```
http://localhost:9000
```



## 🛠 Notes

* Apple Silicon uses `mps` device for acceleration
* Change to `"cuda"` or `"cpu"` if needed
* Large models may take time to download on first run
* Ensure sufficient RAM/VRAM for image models

---


