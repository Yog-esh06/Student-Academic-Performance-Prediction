# Student Academic Performance Prediction

An advanced, research-backed machine learning decision-support framework designed to forecast continuous student examination scores, detect performance risk tiers, quantify prediction uncertainty, and decode model predictions using SHAP (SHapley Additive exPlanations).

---

## 🏛️ Academic Context

* **Institution:** SRM Institute of Science and Technology
* **Subject Name & Code:** Machine Learning - 21CSC305P
* **Mentored by:** [Dr. V. Angayarkanni](https://www.srmist.edu.in/faculty/dr-angayarkanni-v/)
* **Project Team:**
* [Yogesh R Mehta](https://www.linkedin.com/in/yog-esh06/) (RA2411003010776)
* [Bhavy Manchanda](https://www.linkedin.com/in/bhavym/) (RA2411003010749)
* [Darsh M Saraf](https://www.linkedin.com/in/darsh-saraf-540b19323/) (RA2411003010731)
* [Bhavish Akula](https://www.linkedin.com/in/bhavish-akula-4081393b6) (RA2411003010765)



---

## ✨ Key Features

* **Cost-Sensitive Risk Classification:** Implements an asymmetric cost matrix in XGBoost to strictly penalize False Negatives, minimizing false alarms and prioritizing the identification of failing students.
* **Temporal Feature Engineering:** Transforms static dataset snapshots into simulated time-series phases (Early Engagement, Midterm Consistency, Late Term Fatigue) to track student progression.
* **Class Imbalance Resolution (SMOTE):** Utilizes Synthetic Minority Over-sampling Technique to balance the naturally skewed "At-Risk" educational datasets during model training.
* **Prediction Uncertainty Quantification:** Calculates Shannon Entropy on output probabilities. If entropy exceeds 0.85, the system flags the prediction for mandatory human educator review rather than automating a risky intervention.
* **Predictive Regression Pipeline:** Trains and compares 5 continuous machine learning models evaluated via 5-Fold Cross-Validation.
* **Explainable AI (XAI):** Utilizes SHAP interpretability to translate complex mathematical model decisions into clear, human-readable feature impacts alongside temporal phase scores.

---

## 🛠️ Software Architecture & Tech Stack

* **Backend Pipeline:** Python 3.x, Flask (Lightweight WSGI REST API)
* **Machine Learning & Risk Engine:** Scikit-Learn, XGBoost, SHAP, Imbalanced-Learn (SMOTE)
* **Data Manipulation:** Pandas, NumPy
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
│   ├── risk/                     # Cost-sensitive risk engine & temporal feature pipeline
│   └── app/                      # Flask REST API server
│       ├── static/               # Generated EDA images, CSS, inferences.json
│       └── templates/            # HTML frontend (index.html)
├── models_store/                 # Serialized .pkl models, risk classifiers, and preprocessors
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

Open your browser and navigate to `[http://127.0.0.1:5000](http://127.0.0.1:5000)`.

---

## 📊 Model Evaluation Overview

Based on our 5-Fold Cross-Validation metrics across models:

* **Linear Regression (77.0% $R^2$ | RMSE 1.80):** **Winner.** Cleanly maps direct linear relationships (like study hours to scores) with the lowest residual error and highest stability across unseen student subsets.
* **Gradient Boosting (73.5% $R^2$ | RMSE 1.94):** Strong ensemble alternative that sequentially corrects tree errors.
* **XGBoost (67.2% $R^2$ | RMSE 2.15):** Regularized boosted tree framework powering both score expectations and the cost-sensitive risk classification engine.
* **Random Forest (67.1% $R^2$ | RMSE 2.16):** Bagging-based ensemble averaging predictions across independent trees.
* **Decision Tree (45.2% $R^2$ | RMSE 3.54):** Suffers from high variance as it heavily overfits training data with hyper-specific branches.

*A full list of the 10 academic research papers supporting this architecture with functional redirect links can be found in the **References** tab of the application.*

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.