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
st.title("\U0001F393 Student Dropout Risk Predictor")

# Label Mappings (examples - customize with actual labels)
marital_status_map = {
    "Single": 1, "Married": 2, "Widowed": 3, "Divorced": 4, "Separated": 5, "Other": 6
}
application_mode_map = {
    "Online": 1, "In-Person": 2, "Transfer": 3, "Special Program": 4, "Other": 20
}
course_map = {
    "Engineering": 1, "Business": 2, "Law": 3, "Arts": 4, "Computer Science": 5, "Other": 17
}
qualification_map = {
    "None": 1, "High School": 2, "Diploma": 3, "Bachelor's": 4, "Master's": 5, "PhD": 6, "Other": 20
}
occupation_map = {
    "Unemployed": 0, "Clerical": 1, "Skilled Labor": 2, "Professional": 3, "Manager": 4, "Other": 20
}
gender_map = {"Female": 0, "Male": 1}

# Function to take user input
def user_input():
    st.sidebar.subheader("\U0001F4D8 Student Background")
    marital_status = st.sidebar.selectbox("Marital Status", list(marital_status_map.keys()))
    application_mode = st.sidebar.selectbox("Application Mode", list(application_mode_map.keys()))
    application_order = st.sidebar.number_input("Application Order", 1, 6, 1)
    course = st.sidebar.selectbox("Course", list(course_map.keys()))
    attendance = st.sidebar.selectbox("Daytime/Evening Attendance", [0, 1], format_func=lambda x: "Evening" if x == 0 else "Daytime")
    prev_qualification = st.sidebar.selectbox("Previous Qualification", list(qualification_map.keys()))

    st.sidebar.subheader("\U0001F46A Family Info")
    mother_qual = st.sidebar.selectbox("Mother's Qualification", list(qualification_map.keys()))
    father_qual = st.sidebar.selectbox("Father's Qualification", list(qualification_map.keys()))
    mother_occ = st.sidebar.selectbox("Mother's Occupation", list(occupation_map.keys()))
    father_occ = st.sidebar.selectbox("Father's Occupation", list(occupation_map.keys()))

    st.sidebar.subheader("\U0001F3E0 Personal & Financial")
    displaced = st.sidebar.selectbox("Displaced", [0, 1],format_func=lambda x: "No" if x == 0 else "Yes")
    special_needs = st.sidebar.selectbox("Educational Special Needs", [0, 1],format_func=lambda x: "No" if x == 0 else "Yes")
    debtor = st.sidebar.selectbox("Debtor", [0, 1],format_func=lambda x: "No" if x == 0 else "Yes")
    tuition = st.sidebar.selectbox("Tuition Fees Up-to-Date", [0, 1],format_func=lambda x: "No" if x == 0 else "Yes")
    gender = st.sidebar.selectbox("Gender", list(gender_map.keys()))
    scholarship = st.sidebar.selectbox("Scholarship Holder", [0, 1],format_func=lambda x: "No" if x == 0 else "Yes")
    age = st.sidebar.number_input("Age at Enrollment", 17, 62, 20)

    st.sidebar.subheader("\U0001F4DA Academic (Sem 2)")
    sem1_no_eval = st.sidebar.number_input("1st Sem (without evaluations)", 0, 20, 0)
    sem2_cred = st.sidebar.number_input("2nd Sem (credited)", 0, 20, 0)
    sem2_enroll = st.sidebar.number_input("2nd Sem (enrolled)", 0, 20, 0)
    sem2_eval = st.sidebar.number_input("2nd Sem (evaluations)", 0, 20, 0)
    sem2_approved = st.sidebar.number_input("2nd Sem (approved)", 0, 20, 0)
    sem2_grade = st.sidebar.number_input("2nd Sem (grade)", 0.0, 20.0, 10.0)
    sem2_no_eval = st.sidebar.number_input("2nd Sem (without evaluations)", 0, 20, 0)

    st.sidebar.subheader("\U0001F4C9 Economic Indicators")
    unemploy = st.sidebar.number_input("Unemployment Rate", value=-3.12)
    inflation = st.sidebar.number_input("Inflation Rate", value=-0.8)
    gdp = st.sidebar.number_input("GDP", value=13.2)

    data = {
        'Marital status': marital_status_map[marital_status],
        'Application mode': application_mode_map[application_mode],
        'Application order': application_order,
        'Course': course_map[course],
        'Daytime/evening attendance': attendance,
        'Previous qualification': qualification_map[prev_qualification],
        "Mother's qualification": qualification_map[mother_qual],
        "Father's qualification": qualification_map[father_qual],
        "Mother's occupation": occupation_map[mother_occ],
        "Father's occupation": occupation_map[father_occ],
        'Displaced': displaced,
        'Educational special needs': special_needs,
        'Debtor': debtor,
        'Tuition fees up to date': tuition,
        'Gender': gender_map[gender],
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

# Prediction
input_df = user_input()
if st.button("\U0001F50D Predict Dropout Risk"):
    st.subheader("\U0001F4C4 Entered Student Data")
    st.dataframe(input_df)

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    probabilities = model.predict_proba(scaled_input)[0]

    prob_dropout = probabilities[0]
    prob_continue = probabilities[1]

    if prediction == 1:
        st.success(f"🟢 Likely to Continue (Probability: {prob_continue * 100:.2f}%)")
    else:
        st.error(f"🔴 Likely to Dropout (Probability: {prob_dropout * 100:.2f}%)")

    result_data = input_df.copy()
    result_data["Predicted Label"] = "Graduate" if prediction == 1 else "Dropout"
    result_data["Probability (Dropout)"] = f"{prob_dropout * 100:.2f}%"
    result_data["Probability (Continue)"] = f"{prob_continue * 100:.2f}%"

    st.subheader("\U0001F4E5 Download Prediction Result")
    csv = result_data.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download as CSV", csv, "dropout_prediction.csv", "text/csv")

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
    pdf_output = pdf.output(dest='S').encode('latin-1')
    pdf_bytes.write(pdf_output)
    pdf_bytes.seek(0)

    st.download_button(
        label="⬇️ Download as PDF",
        data=pdf_bytes,
        file_name="dropout_prediction.pdf",
        mime="application/pdf"
    )
