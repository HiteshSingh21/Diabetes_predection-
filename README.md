# Diabetes_predection-
A diabetes dataset analysis to derive insightful information and patterns that helps in prediction of diabetes .

# 🩺 Diabetes Prediction using Machine Learning

This repository contains a machine learning project focused on predicting whether a person has diabetes based on diagnostic measurements. The model is built using Python and popular machine learning libraries such as `pandas`, `scikit-learn`, and `matplotlib`.

## 📊 Problem Statement

The goal is to predict the onset of diabetes in patients using medical records that include attributes like glucose level, insulin level, BMI, age, etc. This binary classification task helps in early diagnosis and potential treatment.

## 📁 Project Structure

```

Diabetes\_predection-/
├── dataset/                  # Contains the dataset (e.g., diabetes.csv)
├── Diabetes\_Prediction.ipynb # Jupyter notebook with full analysis and model training
├── requirements.txt          # Required libraries to run the project
├── README.md                 # Project documentation

````

## 🧠 Features Used

The dataset typically includes:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

## 📌 Model Building Steps

1. **Data Cleaning & Preprocessing**
   - Handling missing or zero values
   - Normalizing the data

2. **Exploratory Data Analysis (EDA)**
   - Correlation heatmaps
   - Pair plots and distributions

3. **Model Training**
   - Algorithms used: Logistic Regression, Decision Tree, Random Forest, etc.
   - Cross-validation and hyperparameter tuning

4. **Evaluation Metrics**
   - Accuracy, Precision, Recall, F1-score
   - Confusion matrix and ROC-AUC

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/HiteshSingh21/Diabetes_predection-.git
cd Diabetes_predection-
````

### 2. Create and activate a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate    # On Windows use: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the notebook

Open the Jupyter Notebook to explore data and train models:

```bash
jupyter notebook Diabetes_Prediction.ipynb
```

## ✅ Results

* Best model: Random Forest Classifier (Accuracy: \~85%)
* Feature Importance and ROC Curve visualized
* Model tested with different user inputs

## 📦 Requirements

The project requires the following Python libraries:

```txt
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```

These are included in the `requirements.txt` file.

## 📉 Sample Output (Visualization)

* Confusion Matrix
* Feature Correlation Heatmap
* ROC Curve for Classifiers

## 📌 Future Improvements

* Add more datasets for better generalization
* Deploy model using Flask or Streamlit
* Create a web interface for predictions

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request.
