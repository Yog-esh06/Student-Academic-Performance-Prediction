
# Student Academic Performance Prediction 

An advanced, research-backed machine learning decision-support framework designed to forecast continuous student examination scores, detect performance risk tiers, and decode model predictions using SHAP (SHapley Additive exPlanations). 


---

## 🏛️ Academic Context
* **Institution:** SRM Institute of Science and Technology
* **Subject Name & Code:** Machine Learning - 21CSC305P
* **Mentored by:** [Dr. V. Angayarkanni](https://www.srmist.edu.in/faculty/dr-angayarkanni-v/)
* **Project Team:**
  * [Yogesh R Mehta](https://www.linkedin.com/in/yog-esh06/) (RA2411003010776)
  * [Bhavy Manchanda](https://www.linkedin.com/in/bhavym/) (RA2411003010749)
  * [Darsh M Saraf](https://www.linkedin.com/in/darsh-saraf-540b19323/) (RA2411003010731)
  * Bhavish (RA2411003010765)

---

## ✨ Key Features
* **Cost-Sensitive Risk Classification:** Implements an asymmetric cost matrix in XGBoost to strictly penalize False Negatives, minimizing false alarms and prioritizing the identification of failing students.
* **Temporal Feature Engineering:** Transforms static dataset snapshots into simulated time-series phases (Early Engagement, Midterm Consistency, Late Term Fatigue) to track student progression.
* **Class Imbalance Resolution (SMOTE):** Utilizes Synthetic Minority Over-sampling Technique to balance the naturally skewed "At-Risk" educational datasets during model training.
* **Prediction Uncertainty Quantification:** Calculates Shannon Entropy on output probabilities. If entropy exceeds 0.85, the system flags the prediction for mandatory human educator review rather than automating a risky intervention.
* **Predictive Regression Pipeline:** Trains and compares 5 continuous machine learning models evaluated via 5-Fold Cross-Validation.
* **Explainable AI (XAI):** Utilizes SHAP interpretability to translate complex mathematical model decisions into clear, human-readable feature impacts.

---

## 🛠️ Software Architecture & Tech Stack
* **Backend Pipeline:** Python 3.x
* **Machine Learning:** Scikit-Learn, XGBoost, SHAP
* **Data Manipulation:** Pandas, NumPy
* **API & Server:** Flask (Lightweight WSGI)
* **Frontend:** HTML5, Tailwind CSS, Chart.js, Vanilla JavaScript

---

## 📂 Project Structure
```text
Student-Academic-Performance-Prediction/
├── data/
│   ├── raw/                      # Original dataset (e.g., StudentPerformanceFactors.csv)
│   └── processed/                # Cleaned data outputs
├── src/
│   ├── data/                     # Ingestion, cleaning, EDA generators (eda_generator.py)
│   ├── models/                   # Training, cross-validation, evaluation
│   ├── explainability/           # SHAP global and local attribution logic
│   └── app/                      # Flask REST API server
│       ├── static/               # Generated EDA images, CSS, inferences.json
│       └── templates/            # HTML frontend (index.html)
├── models_store/                 # Serialized .pkl models and preprocessors
├── main.py                       # End-to-end pipeline orchestrator
└── README.md                     # Project documentation

```

---

## 🚀 Setup & Installation

1. **Clone the Repository:**
```bash
git clone https://github.com/Yog-esh06/Student-Academic-Performance-Prediction.git
cd Student-Academic-Performance-Prediction

```


2. **Install Dependencies:**
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt

```


3. **Generate EDA & Train Models (Optional/First Run):**
To build the models and generate the `inferences.json` and EDA graphs from scratch:
```bash
python main.py

```


4. **Launch the Dashboard:**
Start the Flask web server:
```bash
python -m src.app.app

```


Open your browser and navigate to `http://127.0.0.1:5000`.

---

## 📊 Model Evaluation Overview

Based on our 5-Fold Cross-Validation:

* **Linear Regression (Winner):** Cleanly maps the direct linear relationships (like study hours to scores), yielding the lowest RMSE (1.80) and highest $R^2$ (77.0%).
* **Gradient Boosting & XGBoost:** Strong ensemble performers but slightly underfit due to the continuous linear nature of this specific dataset.
* **Decision Trees:** Suffer from high variance (RMSE 3.53) as they heavily overfit the training data by creating hyper-specific branches.

*A full list of the 9 academic research papers supporting this architecture can be found in the **References** tab of the application.*

## 👤 Author

Yogesh R Mehta

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.


