# 🌸 Femina-AI

**Femina-AI** is a machine learning-powered application designed to assist in the prediction and analysis of Polycystic Ovary Syndrome (PCOS).
Built with a Python/Flask backend, this project utilizes advanced ML models alongside Explainable AI (SHAP) to provide transparent, 
data-driven medical insights.

---

## ✨ Features

* **PCOS Risk Prediction:** Leverages trained machine learning models to assess PCOS probability based on patient data metrics.
* **Explainable AI (XAI):** Integrates SHAP (SHapley Additive exPlanations) to explain model predictions, providing transparency for medical professionals and users.
* **Doctor Service Module:** Includes built-in services (`doctor_service.py`) for handling medical professional interactions or specific clinical logic.
* **Database Management:** Uses SQLite for lightweight, efficient data storage with automated scripts for resetting and managing the database.
* **Deployment Ready:** Comes pre-configured for containerized deployment with `Dockerfile` and serverless deployment with `vercel.json`.

---

## 🛠️ Technology Stack

* **Backend:** Python 3.x, Flask (`flaskapp.py`)
* **Machine Learning:** scikit-learn, SHAP (`create_shap_explainer.py`), Pandas
* **Database:** SQLite (`pcos_db.db`)
* **Deployment:** Docker, Vercel
* **Package Management:** `requirements.txt`, `pyproject.toml`

---

## 📁 Project Structure

```text
Femina-AI/
├── .env.example                 # Environment variables template
├── Dockerfile                   # Docker container configuration
├── vercel.json                  # Vercel deployment configuration
├── flaskapp.py                  # Main Flask application instance
├── run_server.py                # Entry point to start the web server
├── doctor_service.py            # Service logic for medical/doctor interactions
├── train_model.py               # Script to train the core ML model
├── create_shap_explainer.py     # Script to generate SHAP values for model explainability
├── pcos_synthetic_50000.csv     # Synthetic dataset containing 50,000 PCOS records
├── pcos_db.db                   # SQLite database
├── reset_db.py                  # Utility script to reset the database state
├── debug_validation.py          # Debugging tool for data validation
├── debug_verify_model_load.py   # Debugging tool to verify ML model loading
├── test_predict.py              # Script to test inference and predictions
├── requirements.txt             # Python dependencies
└── pyproject.toml               # Python project build requirements

```

---

## 🚀 Getting Started

### Prerequisites

* Python 3.8+ installed on your local machine.
* `pip` (Python package manager).

### Installation & Setup

1. **Clone the repository:**
```bash
git clone [https://github.com/your-username/femina-ai.git](https://github.com/your-username/femina-ai.git)
cd femina-ai

```


2. **Set up a virtual environment (Recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```


4. **Configure Environment Variables:**
* Copy the `.env.example` file to a new file named `.env`.
* Fill in any necessary API keys or secret configurations.


```bash
cp .env.example .env

```


5. **Run the Application:**
```bash
python run_server.py

```


*The server should now be running locally (usually on `http://localhost:5000` or `http://127.0.0.1:5000`).*

---

## 🧠 Machine Learning & Data Pipeline

If you want to retrain the model or test the data pipeline locally:

1. **Dataset:** The project utilizes a synthetic dataset of 50,000 records (`pcos_synthetic_50000.csv`).
2. **Train the Model:** Run `python train_model.py` to generate the latest model artifacts.
3. **Generate Explainer:** Run `python create_shap_explainer.py` to update the Explainable AI SHAP components.
4. **Test Predictions:** Use `python test_predict.py` to verify that the newly trained model outputs the expected results.

---

## 🌐 Deployment

### Using Docker

You can containerize the application for deployment anywhere using the provided Dockerfile:

```bash
docker build -t femina-ai .
docker run -p 5000:5000 femina-ai

```

### Using Vercel

The project includes a `vercel.json` file, making it ready for immediate deployment on Vercel's serverless infrastructure. Simply connect your GitHub repository to Vercel and it will automatically deploy based on these configurations.

```

```
