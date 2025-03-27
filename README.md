# ❤️ Heart Disease Prediction

This project predicts the likelihood of heart disease based on patient health data using logistic regression. It includes data cleaning, visual analysis using Python and SQL, and interprets feature influence.

## 📊 Project Overview

- Cleaned and prepared real-world heart disease data
- Performed exploratory data analysis (EDA) using Python and SQL
- Created visualizations to understand trends and correlations
- Built a logistic regression model to predict heart disease severity
- Identified features with the strongest influence on predictions

## 🧠 Technologies Used

- **Python** (pandas, matplotlib, seaborn, scikit-learn)
- **SQLite** / SQL
- **VS Code** & Terminal
- **Git** & GitHub

## 📁 Project Structure

```
data/                 <- Cleaned dataset (heart.csv)
src/                  <- Python scripts for EDA, modeling, SQL plotting
queries/              <- SQL EDA queries
eda_visualizations/   <- Charts generated from SQL and Python
heart_disease.db      <- SQLite database
README.md             <- You are here
```

## 📊 Results Summary

- ✅ **Model Accuracy**: 61.6%
- ✅ **Most influential feature**: Number of major vessels (`ca`)
- ✅ **Highest heart disease rate** in patients with cholesterol > 280

## 🚀 How to Run This Project

1. **Clone this repository**:

   ```bash
   git clone https://github.com/suhaimibira/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run Python scripts**:

   ```bash
   python src/clean_data.py         # Clean the raw data
   python src/eda.py                # Create visualizations using Python
   python src/logistic_model.py     # Train and evaluate logistic regression model
   python src/run_sql_eda.py        # Run SQL-based analysis and charts
   ```

## 🙌 Acknowledgments

- Data Source: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)

---

Made with ❤️ by **Sabirah** | [GitHub Profile](https://github.com/suhaimibira)

