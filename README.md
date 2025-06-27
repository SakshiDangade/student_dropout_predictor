# 🎓 Student Dropout Risk Predictor

This project is a **machine learning-powered web app** designed to predict whether a student is likely to **drop out or graduate** based on academic, demographic, and socio-economic features.

It is built with **Python**, trained in **Google Colab**, and deployed using **Streamlit**.

---

## 🚀 Live Demo

🔗 [Try the Live App](https://studentdropoutpredictor-q63hpgiwx9umxogizth8m4.streamlit.app/)

---

## 🧠 Models Trained

Four machine learning models were trained and compared:

- ✅ Logistic Regression *(Best Performer)*
- 🌲 Decision Tree Classifier
- 📊 K-Nearest Neighbors
- 🌳 Random Forest Classifier

After testing with accuracy, precision, and recall, **Logistic Regression** performed best and was selected for deployment.

---

## 📂 Project Structure

Student-Dropout-Predictor-App/
│
├── models/
│ ├── dropout_model.pkl # Trained logistic regression model
│ └── scaler.pkl # Feature scaler used during preprocessing
│
├── app.py # Main Streamlit web app
├── requirements.txt # Python dependencies
└── README.md # Project documentation (this file)

---

## 📊 Features

- User-friendly web interface (via Streamlit)
- Predicts student dropout or graduation
- Displays class probabilities
- Allows **CSV** and **PDF downloads** of the prediction
- Clean sidebar UI with dropdowns for better user clarity

---

## 📈 Input Features

The model uses 30+ features, including:

- Demographic data (e.g., gender, age, marital status)
- Academic records (credits, grades, evaluations)
- Parental education and occupation
- Socio-economic indicators (GDP, inflation, unemployment)
- Tuition payment status and scholarships

---

🧾 Sample Prediction Output

The app displays the predicted class and probabilities like:

🟢 Likely to Graduate (Probability: 82.15%)
🔴 Likely to Dropout (Probability: 17.85%)

And allows downloading the result as:

-CSV File
-PDF Report

📚 Learnings

-Model comparison and evaluation
-Handling label-encoded categorical features
-Model serialization using joblib
-Real-time web deployment with Streamlit
-Improving user experience with readable labels and dropdowns

