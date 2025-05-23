# 🎙️ Speech Processing Backend

A **FastAPI**-based backend service designed for advanced speech processing tasks leveraging **OpenAI Whisper**, **Transformers**, and deep learning models for multimodal input processing, NLP, and generative response creation. Hosted on a **Google Cloud Platform (GCP) VM**, this project features automated CI/CD pipelines with robust code quality and security tooling.

---

## 🧠 Project Overview

The backend ingests raw audio data, transcribes it into text, and processes this text through sophisticated NLP pipelines. Additionally, it integrates facial landmark analysis for multimodal input (voice + face) applications, enabling a richer contextual understanding and advanced functionalities such as:

* **Speech Recognition**: Using OpenAI’s Whisper for state-of-the-art voice-to-text transcription
* **Facial Landmark Detection**: Align faces and extract lip coordinates with Dlib and face-alignment tools to improve speech analysis
* **NLP Pipelines**: Including emotion recognition, phoneme detection, and text generation via HuggingFace Transformers
* **Audio & Video Processing**: Leveraging `librosa`, `moviepy`, and `torch` for advanced feature extraction and model input preparation

This architecture supports various use cases from voice assistants, multimodal chatbots, to emotion-aware interactive agents.

---

## ⚙️ Technology Stack

| Layer                           | Technologies & Libraries                                                 |
| ------------------------------- | ------------------------------------------------------------------------ |
| **Backend API**                 | FastAPI, Uvicorn, Starlette                                              |
| **Speech Recognition**          | OpenAI Whisper, Editdistance                                             |
| **Audio Processing**            | Librosa, FFmpeg, Soundfile, Audioread, Soxr                              |
| **NLP / ML Models**             | HuggingFace Transformers, NLTK, Scikit-learn, TensorFlow, Keras, PyTorch |
| **Face Analysis**               | face-alignment, dlib, OpenCV                                             |
| **Data Handling & Performance** | NumPy, SciPy, tqdm, Numba, llvmlite                                      |
| **Security & Auth**             | cryptography, oauthlib, requests-oauthlib, python-dotenv                 |
| **Quality & Security**          | SonarQube, Snyk, Pylint, Black                                           |
| **Deployment**                  | Docker, GCP VM (Ubuntu 24.10), GitHub Actions                            |

---

## ☁️ Deployment Architecture

* **Hosting:** Google Cloud VM (e2-standard-4, 4 vCPUs, 16 GB RAM, 100 GB SSD), located in Singapore (`asia-southeast1-a`).
* **OS:** Ubuntu 24.10 Minimal for lightweight and optimized environment.
* **CI/CD:** Automated workflows in GitHub Actions handle:

  * Dependency installation
  * Static code analysis (SonarQube) and security vulnerability checks (Snyk)
  * Containerization & deployment to the GCP VM via SSH commands
* **Monitoring:** Integrated quality gates and security alerts help maintain codebase health continuously.

---

## 🏗️ System Architecture Overview

```plaintext
Audio Input (client) 
    ↓
API Endpoint (/predict) receives audio file → 
    ↓
Preprocessing: Audio cleanup & feature extraction (librosa, ffmpeg) → 
    ↓
Speech-to-Text (OpenAI Whisper) → 
    ↓
Optional Multimodal Input: Face alignment & lip coordinate extraction (dlib, face-alignment) → 
    ↓
NLP Pipeline: emotion recognition, phoneme detection, text generation (Transformers, PyTorch) → 
    ↓
Response JSON → Delivered to client
```

---

## 📁 Project Structure

```
📦test-sp-be
 ┣ 📂.github
 ┃ ┗ 📂workflows                                    # CI/CD pipelines configs
 ┃   ┣ 📄cicd.yaml
 ┃   ┣ 📄deploy.yaml
 ┃   ┣ 📄snyk-scan.yaml
 ┃   ┗ 📄sonarqube-analysis.yaml
 ┣ 📂lip_coordinate_extraction                      # Face landmark extraction utils
 ┃ ┣ 📄lips_coords_extractor.py
 ┃ ┗ 📄shape_predictor_68_face_landmarks_GTX.dat
 ┣ 📂pretrain                                      # Pretrained model weights
 ┃ ┗ 📄LipCoordNet_coords_loss_0.02558115310966...
 ┣ 📄.dockerignore
 ┣ 📄.gitattributes
 ┣ 📄.gitignore
 ┣ 📄Dockerfile                                     # Docker container setup
 ┣ 📄app.py                                         # FastAPI app entrypoint
 ┣ 📄cvtransforms.py                                # Custom CV transforms
 ┣ 📄dataset.py                                     # Dataset utilities
 ┣ 📄inference.py                                   # Inference pipeline implementation
 ┣ 📄README.md
 ┣ 📄readme.txt
 ┣ 📄requirements.txt                               # Python dependencies
 ┣ 📄sonar-project.properties                       # SonarQube configuration
```

---

## 🚀 Running Locally

Ensure you have Python 3.11+ installed and all dependencies:

```bash
git clone https://github.com/rp-org/test-sp-be.git
cd test-sp-be
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

---

## ✅ API Endpoints & Usage

### 1. **Health Check**

Check server status:

```http
GET https://35.247.158.101/test-sp-be/health
```

Expected response:

```json
{
    "status": "online",
    "service": "Speech Processing Backend",
    "message": "The backend is running and ready to process video/audio for speech recognition and lip-reading."
}
```

---

### 2. **Speech Processing Prediction**

Submit an audio file for processing:

```http
POST https://35.247.158.101/test-sp-be/predict
Content-Type: multipart/form-data
```

**Request body:**

* `file`: Audio file (wav, mp3, etc.)

**Curl example:**

```bash
curl -X POST "https://35.247.158.101/test-sp-be/predict" \
  -F "file=@/path/to/your/audio.wav"
```

**Response:**

```json
{
    "expected_text": "\"Helo\"",
    "mode": "\"sentence\"",
    "whisper_text": " கணகாயம் பரையோ சையி ஒரு வாழ்வின் பிரம்மாகிடுமே என்னுமே மலராதனை நிஞ்சம் மேங்கே வணிப்பிக்கும் உன் கண்ணில் முற்காகிடுமே என்னுமே நீ பாராயாம்",
    "whisper_accuracy": 0.0,
    "wav2vec_text": "ON A GA YOU BUT A YORS THE BOD PI IN  BE TAM  PA YE TO P AN BEN MY LAD A DELA DIN TA IN BBUDY B GUN WUN GUNIE  BOD GAN A A G GOO DE TO PAN BE B B A A",
    "wav2vec_accuracy": 5.161290322580648,
    "lip_reading_text": "ON A GA YOU BUT A YORS THE BOD PI IN  BE TAM  PA YE TO P AN BEN MY LAD A DELA DIN TA IN BBUDY B GUN WUN GUNIE  BOD GAN A A G GOO DE TO PAN BE B B A A",
    "lip_reading_accuracy": 5.161290322580648,
    "overall_accuracy": 2.580645161290324,
    "feedback": "Needs more practice !"
}
```

---

### 3. **Swagger UI**

Interactively test API endpoints with Swagger docs:

```
https://35.247.158.101/test-sp-be/docs
```

---

## 🔐 Security & Monitoring

* All sensitive secrets (e.g., API tokens, SSH keys) are managed via GitHub Actions secrets.
* Snyk performs continuous vulnerability scans on dependencies.
* SonarQube runs automated code quality and technical debt analysis on every PR and push.

---

## 🧪 CI/CD Pipelines (GitHub Actions)

### SonarQube Analysis

* Checks out code
* Installs dependencies
* Runs SonarQube scan and blocks merge if Quality Gate fails

### Snyk Scan

* Performs static dependency vulnerability scan
* Reports issues directly in PR comments

### Deployment

* SSH into GCP VM
* Pulls latest branch changes
* Rebuilds and restarts FastAPI backend service

---

## 📦 Docker Support

A `Dockerfile` is included for containerizing the app:

```bash
docker build -t test-sp-be .
docker run -p 8000:8000 test-sp-be
```

Use containers for consistent deployment environments and easy scalability.

---

## 🔗 Repository

Find the full source code and issue tracker here:
[https://github.com/rp-org/test-sp-be](https://github.com/rp-org/test-sp-be)

---

## 💡 Contribution Guidelines

Want to add new features like:

* Additional NLP models?
* Multi-language support?
* Enhanced emotion detection?

Feel free to fork, make your changes, and submit a pull request. Please adhere to the code style and include tests where applicable.

---

## 📜 License

This project is licensed under the MIT License — free to use, modify, and distribute.


