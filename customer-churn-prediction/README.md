
# 📊 Customer Churn Prediction System

A Machine Learning web application that predicts customer churn risk using customer profile and service information. Built with Python, Scikit-learn, Pandas, Plotly, and Streamlit.

## 🚀 Features

- **Individual Prediction:** Enter customer details to estimate churn probability.
- **Analytics Dashboard:** Explore customer churn patterns through interactive charts and KPIs.
- **Bulk Prediction:** Upload a CSV file to predict churn for multiple customers.
- **Downloadable Results:** Export bulk predictions as a CSV file.
- **Model Evaluation:** Review accuracy, precision, recall, F1 score, ROC-AUC, and a confusion matrix.
- **Risk Guidance:** Display illustrative churn-risk levels and suggested retention actions.

## 🛠️ Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Plotly
- Matplotlib and Seaborn
- Joblib

## 📁 Project Structure

```text
customer-churn-prediction/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── models/
    ├── churn_model.pkl
    └── model_columns.pkl
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd customer-churn-prediction
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🤖 Machine Learning

The project uses a Scikit-learn classification pipeline with preprocessing for numerical and categorical features.

Logistic Regression was selected during model comparison against Decision Tree and Random Forest models.

The model provides:
- Binary churn prediction
- Estimated churn probability
- Classification metrics for model evaluation

## 📈 Model Performance

The reported test-set results from the initial training run were:

| Metric | Score |
|---|---:|
| Accuracy | 80.55% |
| Precision | 65.72% |
| Recall | 55.88% |
| F1 Score | 60.40% |
| ROC-AUC | 0.8419 |

These are the original training-run results. The evaluation section in the app may show different values if it uses a different test split.

## 📊 Dataset

This project uses the Telco Customer Churn dataset, containing customer demographics, subscribed services, account information, and churn status.

Check the dataset's original source and license before redistributing it.

## ⚠️ Disclaimer

Predictions are estimates based on historical data and are not guarantees of future customer behavior. Risk categories are illustrative and should be validated before being used for business decisions.

## 👨‍💻 Author

**Onkar Wakchaure**

GitHub: Add your GitHub profile URL here.

Portfolio: Add your portfolio URL here.

LinkedIn: Add your LinkedIn profile URL here.
