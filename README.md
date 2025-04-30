# 💳 FRAUDULENT TRANSACTION DETECTION

**End-to-End Machine Learning Pipeline for Financial Fraud Detection**

This project aims to detect fraudulent financial transactions using an end-to-end machine learning pipeline. It handles real-world challenges such as imbalanced datasets, domain-specific feature engineering, model evaluation, and explainability. The core model is powered by XGBoost, known for its speed and accuracy in structured/tabular data.

---

## 🧠 Project Objectives

- Build a supervised machine learning classifier to identify fraudulent transactions
- Handle extreme class imbalance using SMOTE
- Engineer features that reflect transaction anomalies
- Evaluate model performance using appropriate metrics (confusion matrix, ROC-AUC)
- Provide interpretability using SHAP to explain predictions to stakeholders

---

## 🔧 Tech Stack

| Category           | Tools & Libraries                              |
|--------------------|-------------------------------------------------|
| Language           | Python                                          |
| IDE                | Jupyter Notebook / VS Code                      |
| Data Handling      | pandas, numpy                                   |
| Modeling           | scikit-learn, XGBoost                           |
| Imbalance Handling | imbalanced-learn (SMOTE)                        |
| Visualization      | matplotlib, seaborn                             |
| Explainability     | SHAP                                            |

---

## ⚙️ Pipeline Architecture


---

## 📌 Key Highlights

- ✅ Built a machine learning pipeline to classify fraudulent financial transactions with **95%+ accuracy**
- 🛠️ Performed **domain-specific feature engineering** to enhance signal quality
- ⚖️ Handled severe class imbalance using **SMOTE oversampling and threshold tuning**
- 📊 Visualized model performance with **confusion matrices**, **ROC curves**, and **classification reports**
- 🔍 Used **SHAP values** to explain individual predictions and global feature importance

---

## 📁 Project Structure


---

## 📊 Evaluation Metrics

- **Accuracy**
- **Precision, Recall, F1-score**
- **ROC AUC Score**
- **Confusion Matrix**

These metrics help measure how well the model performs in a high-class-imbalance environment where simply predicting "non-fraud" all the time isn't acceptable.

---

## 🔎 Explainability with SHAP

To gain trust and ensure transparency in predictions, **SHAP (SHapley Additive exPlanations)** is used to:
- Visualize which features contributed most to fraud classification
- Provide global and local feature attributions
- Assist stakeholders in understanding model behavior

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/IEncryptSaad/FRAUDULENT-TRANSACTION-DETECTION.git
   cd FRAUDULENT-TRANSACTION-DETECTION
## 🔍 Load Trained Model

To load and use the trained XGBoost fraud detection model:

```python
import joblib
model = joblib.load('notebook/xgboost_fraud_model.model')
## 🚀 Model Summary

After loading the `.model` file, others can:

- 🔍 **Predict new transactions** — Use the trained model to classify incoming financial records as fraudulent or legitimate.
- 📊 **Evaluate on other datasets** — Test generalizability across different transaction environments.
- 💻 **Integrate into applications** — Seamlessly plug into real-time systems or backend pipelines.
- 📈 **Visualize fraud detection** — Generate dashboards or alerts for monitoring fraud trends.

---

## 🌐 3. Use in a Web App or API

You can embed this model into various interactive platforms:

- ⚙️ **Flask API** — Serve real-time fraud predictions via REST endpoints.
- 🧪 **Streamlit app** — Build a GUI to test transactions live.
- 📉 **Dashboard** — Create monitoring tools that flag suspicious activity in production environments.

---

## 🛠️ 4. Continue Training or Fine-tune

While `.pkl` models are not typically re-trained, XGBoost allows:

- Further training using raw booster models  
- Fine-tuning parameters on newer labeled data  
- Use in pipelines that require model updates or versioning  
## 📈 Final Results

- ✅ Accuracy: 99.80%
- ✅ Best Model: Tuned XGBoost
- ✅ SHAP used for explainability
- ✅ Model saved as `.model` and usable via `joblib.load()`
