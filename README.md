# 🎵 Music Recommender System

A production-oriented music recommendation system that delivers similar song suggestions using cosine similarity on text-based features, enhanced with Spotify API integration for album artwork.

---

## 🚀 Overview

This project implements a **complete ML pipeline + serving layer**:

* Data preprocessing and feature extraction from song metadata
* Vector space modeling using `CountVectorizer`
* Cosine similarity–based recommendation engine
* Streamlit UI for interactive inference
* Spotify API integration for enriched user experience

The system is designed with a focus on **reproducibility, robustness, and clean engineering practices**.

---

## 🧠 Core Features

* 🔍 Content-based song recommendations (top-N similar tracks)
* 🎧 Dynamic album artwork via Spotify API
* ⚡ Fast inference using precomputed similarity matrix
* 🛡️ Secure configuration via environment variables
* 🧩 Graceful degradation when data/API is unavailable
* 🧪 Reproducible training pipeline (`train.py` + notebook)

---

## 🏗️ Architecture

```id="arch01"
          ┌─────────────────────────┐
          │   Training Pipeline     │
          │  (train.py / notebook)  │
          └──────────┬──────────────┘
                     │
         Generates df.pkl + similarity.pkl
                     │
          ┌──────────▼──────────┐
          │   Streamlit App     │
          │      (app.py)       │
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │   Spotify API       │
          │ (album artwork)     │
          └─────────────────────┘
```

---

## 📂 Project Structure

```id="struct01"
MUSIC-recommendation-system/
│
├── app.py                  # Streamlit inference layer
├── train.py                # CLI-based training pipeline
├── requirements.txt        # Dependencies
├── .env.example            # Environment config template
├── README.md               # Documentation
├── Model Training.ipynb    # Experimental pipeline (cleaned)
│
└── (generated)
    ├── df.pkl              # Processed dataset
    └── similarity.pkl      # Similarity matrix
```

---

## ⚙️ Setup & Installation

### 1. Clone repository

```bash id="cmd01"
git clone https://github.com/ARYANBHAT-eng/MUSIC-recommendation-system.git
cd MUSIC-recommendation-system
```

---

### 2. Create virtual environment

```bash id="cmd02"
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

---

### 3. Install dependencies

```bash id="cmd03"
pip install -r requirements.txt
```

---

### 4. Configure environment variables

```bash id="cmd04"
cp .env.example .env
```

Edit `.env`:

```env id="env01"
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
```

---

## 🧪 Training the Model

You can generate artifacts using:

```bash id="cmd05"
python train.py --input spotify_millsongdata.csv --output .
```

This will produce:

* `df.pkl`
* `similarity.pkl`

---

## ▶️ Running the Application

```bash id="cmd06"
streamlit run app.py
```

---

## 📦 Data & Artifacts

Required runtime files:

* `df.pkl` → processed dataset
* `similarity.pkl` → cosine similarity matrix

> These are not included in the repository due to size constraints.

---

## ⚠️ Limitations

* Uses full NxN similarity matrix (memory-bound for large datasets)
* No online/real-time model updates
* Depends on Spotify API availability for artwork

---

## 🧱 Future Improvements

* Replace dense similarity with ANN / top-K retrieval
* Introduce model versioning and artifact storage
* Add REST API layer (FastAPI) for serving
* Implement caching for external API calls
* Deploy on cloud (Streamlit Cloud / AWS / GCP)

---

## 🧑‍💻 Author

**Aryan Bhat**

* GitHub: https://github.com/ARYANBHAT-eng
* LinkedIn: https://linkedin.com/in/AryanBhat

---

## 📄 License

MIT License

---

## ⭐ Final Note

This project demonstrates:

> End-to-end ML system design — from data processing to model serving — with an emphasis on clean architecture, reproducibility, and practical engineering trade-offs.
