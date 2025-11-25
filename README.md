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
│
├── templates/
│   └── index.html
│
├── app.py
├── requirements.txt
└── README.md
```

---

## ▶️ Running the Application

Launch the Flask server:

```bash
python app.py
```

By default, the server runs on:

```
http://0.0.0.0:9000
```

Open in your browser:

```
http://localhost:9000
```

This loads the UI for all three modules:

* **Image Explainer**
* **Sentiment Analyzer**
* **Dialogue Summarizer**

---

# 🌐 API Endpoints

---

## 1️⃣ `/sentiment_analysis` — **GET**

Runs sentiment analysis on the supplied text.

### **Query Parameter**

| Name            | Type   | Required | Description |
| --------------- | ------ | -------- | ----------- |
| `textToAnalyze` | string | yes      | Input text  |

### Example

```
GET /sentiment_analysis?textToAnalyze=I love this!
```

#### Response

```json
{
  "label": "POSITIVE",
  "score": 0.97
}
```

---

## 2️⃣ `/image_analysis` — **POST**

Analyzes an uploaded image and returns a detailed description.

### **Form-Data Fields**

| Field    | Type   | Required | Description        |
| -------- | ------ | -------- | ------------------ |
| `image`  | file   | yes      | Image to analyze   |
| `prompt` | string | no       | Custom instruction |

### Example (cURL)

```bash
curl -X POST http://localhost:9000/image_analysis \
  -F "image=@example.jpg" \
  -F "prompt=Describe this image."
```

#### Response

```json
{
  "answer": "The image shows three hikers walking..."
}
```

---

## 

## 🛠 Notes

* Apple Silicon uses `mps` device for acceleration
* Change to `"cuda"` or `"cpu"` if needed
* Large models may take time to download on first run
* Ensure sufficient RAM/VRAM for image models

---


