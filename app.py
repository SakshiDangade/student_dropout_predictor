import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
from fpdf import FPDF

# Load model and scaler
model = joblib.load("models/dropout_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Page settings
st.set_page_config(page_title="Student Dropout Predictor", layout="centered")
st.title("🎓 Student Dropout Risk Predictor")

# Function to take user input
def user_input():
    st.sidebar.subheader("📘 Student Background")
    marital_status = st.sidebar.number_input("Marital Status", 1, 6, 1)
    application_mode = st.sidebar.number_input("Application Mode", 1, 20, 1)
    application_order = st.sidebar.number_input("Application Order", 1, 6, 1)
    course = st.sidebar.number_input("Course ID", 1, 17, 1)
    attendance = st.sidebar.selectbox("Daytime/Evening Attendance", [0, 1], format_func=lambda x: "Evening" if x == 0 else "Daytime")
    prev_qualification = st.sidebar.number_input("Previous Qualification", 1, 20, 1)

    st.sidebar.subheader("👪 Family Info")
    mother_qual = st.sidebar.number_input("Mother's Qualification", 1, 30, 10)
    father_qual = st.sidebar.number_input("Father's Qualification", 1, 30, 3)
    mother_occ = st.sidebar.number_input("Mother's Occupation", 0, 20, 0)
    father_occ = st.sidebar.number_input("Father's Occupation", 0, 20, 0)

    st.sidebar.subheader("🏠 Personal & Financial")
    displaced = st.sidebar.selectbox("Displaced", [0, 1])
    special_needs = st.sidebar.selectbox("Educational Special Needs", [0, 1])
    debtor = st.sidebar.selectbox("Debtor", [0, 1])
    tuition = st.sidebar.selectbox("Tuition Fees Up-to-Date", [0, 1])
    gender = st.sidebar.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    scholarship = st.sidebar.selectbox("Scholarship Holder", [0, 1])
    age = st.sidebar.number_input("Age at Enrollment", 17, 62, 20)

    st.sidebar.subheader("📚 Academic (Sem 2)")
    sem1_no_eval = st.sidebar.number_input("1st Sem (without evaluations)", 0, 20, 0)
    sem2_cred = st.sidebar.number_input("2nd Sem (credited)", 0, 20, 0)
    sem2_enroll = st.sidebar.number_input("2nd Sem (enrolled)", 0, 20, 0)
    sem2_eval = st.sidebar.number_input("2nd Sem (evaluations)", 0, 20, 0)
    sem2_approved = st.sidebar.number_input("2nd Sem (approved)", 0, 20, 0)
    sem2_grade = st.sidebar.number_input("2nd Sem (grade)", 0.0, 20.0, 10.0)
    sem2_no_eval = st.sidebar.number_input("2nd Sem (without evaluations)", 0, 20, 0)

    st.sidebar.subheader("📉 Economic Indicators")
    unemploy = st.sidebar.number_input("Unemployment Rate", value=-3.12)
    inflation = st.sidebar.number_input("Inflation Rate", value=-0.8)
    gdp = st.sidebar.number_input("GDP", value=13.2)

    data = {
        'Marital status': marital_status,
        'Application mode': application_mode,
        'Application order': application_order,
        'Course': course,
        'Daytime/evening attendance': attendance,
        'Previous qualification': prev_qualification,
        "Mother's qualification": mother_qual,
        "Father's qualification": father_qual,
        "Mother's occupation": mother_occ,
        "Father's occupation": father_occ,
        'Displaced': displaced,
        'Educational special needs': special_needs,
        'Debtor': debtor,
        'Tuition fees up to date': tuition,
        'Gender': gender,
        'Scholarship holder': scholarship,
        'Age at enrollment': age,
        'Curricular units 1st sem (without evaluations)': sem1_no_eval,
        'Curricular units 2nd sem (credited)': sem2_cred,
        'Curricular units 2nd sem (enrolled)': sem2_enroll,
        'Curricular units 2nd sem (evaluations)': sem2_eval,
        'Curricular units 2nd sem (approved)': sem2_approved,
        'Curricular units 2nd sem (grade)': sem2_grade,
        'Curricular units 2nd sem (without evaluations)': sem2_no_eval,
        'Unemployment rate': unemploy,
        'Inflation rate': inflation,
        'GDP': gdp
    }

    return pd.DataFrame([data])

# Get user input
input_df = user_input()

# Predict button
if st.button("🔍 Predict Dropout Risk"):
    st.subheader("📄 Entered Student Data")
    st.dataframe(input_df)

    # Scale input
    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]
    probabilities = model.predict_proba(scaled_input)[0]

    prob_dropout = probabilities[0]     # Class 0 = Dropout
    prob_continue = probabilities[1]    # Class 1 = Graduate

    # Show result
    if prediction == 1:
        st.success(f"🟢 Likely to Continue (Probability: {prob_continue * 100:.2f}%)")
    else:
        st.error(f"🔴 Likely to Dropout (Probability: {prob_dropout * 100:.2f}%)")

    # Prepare result data
    result_data = input_df.copy()
    result_data["Predicted Label"] = "Graduate" if prediction == 1 else "Dropout"
    result_data["Probability (Dropout)"] = f"{prob_dropout * 100:.2f}%"
    result_data["Probability (Continue)"] = f"{prob_continue * 100:.2f}%"

    st.subheader("📥 Download Prediction Result")

    # CSV
    csv = result_data.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download as CSV", csv, "dropout_prediction.csv", "text/csv")

    # PDF
    class PDF(FPDF):
        def header(self):
            self.set_font("Arial", "B", 14)
            self.cell(0, 10, "Student Dropout Risk Prediction Report", ln=True, align="C")
            self.ln(10)

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for col, val in result_data.iloc[0].items():
        pdf.cell(0, 10, f"{col}: {val}", ln=True)

    pdf_bytes = io.BytesIO()
    pdf_output = pdf.output(dest='S').encode('latin-1')  # Output as string, encode to bytes
    pdf_bytes.write(pdf_output)
    pdf_bytes.seek(0)

    st.download_button(
        label="⬇️ Download as PDF",
        data=pdf_bytes,
        file_name="dropout_prediction.pdf",
        mime="application/pdf"
    )
