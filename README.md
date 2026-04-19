# 🎵 Music Recommender System

A Streamlit-based music recommendation engine that suggests similar songs using cosine similarity on song metadata, enhanced with Spotify API integration for album artwork.

---

## 🚀 Overview

This project demonstrates an end-to-end recommendation workflow:

* Data preprocessing and feature engineering (via training notebook)
* Similarity computation using vector space modeling
* Real-time recommendations through a Streamlit UI
* External API integration (Spotify) for enriched user experience

The application is designed with modularity, fault tolerance, and clean engineering practices.

---

## 🧠 Features

* 🔍 Select a song and get top-N similar recommendations
* 🎧 Spotify-powered album artwork
* ⚡ Fast lookup using precomputed similarity matrix
* 🛡️ Secure API handling via environment variables
* 🧩 Graceful fallback when data or API is unavailable

---

## 🏗️ Tech Stack

* **Python**
* **Streamlit**
* **Pandas / NumPy**
* **Scikit-learn**
* **Spotipy (Spotify API)**

---

## 📂 Project Structure

```
MUSIC-recommendation-system/
│
├── app.py                  # Main Streamlit application
├── requirements.txt       # Project dependencies
├── .env.example           # Environment variable template
├── README.md              # Project documentation
└── Model Training.ipynb   # (Optional) Model training pipeline
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/ARYANBHAT-eng/MUSIC-recommendation-system.git
cd MUSIC-recommendation-system
```

---

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Configure environment variables

Copy the example file:

```bash
cp .env.example .env
```

Update `.env` with your credentials:

```
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
```

---

### 5. Run the application

```bash
streamlit run app.py
```

---

## 📦 Data & Model Artifacts

This project relies on precomputed files:

* `df.pkl` → processed song dataset
* `similarity.pkl` → cosine similarity matrix

⚠️ These files are not included in the repository due to size constraints.

### To generate them:

* Use `Model Training.ipynb`
* Or integrate your own dataset pipeline

---

## ⚠️ Known Limitations

* Requires precomputed similarity matrix
* Spotify API usage depends on valid credentials
* No live model retraining in current version

---

## 🧱 Future Improvements

* Replace pickle-based storage with scalable model serving
* Add caching layer for API responses
* Deploy on cloud (Streamlit Cloud / AWS / GCP)
* Integrate real-time recommendation updates

---

## 🧑‍💻 Author

**Aryan Bhat**

* GitHub: https://github.com/ARYANBHAT-eng
* LinkedIn: https://linkedin.com/in/AryanBhat

---

## 📄 License

This project is open-source and available under the MIT License.

---

## ⭐ Final Note

This project is built as a scalable prototype demonstrating:

> Clean engineering practices, modular design, and applied machine learning in a real-world use case.
