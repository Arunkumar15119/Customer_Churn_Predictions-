import streamlit as st
import pickle
import pandas as pd

# --->Loading the trained model and scaler 
model_path = 'RF_model.pkl'  
scaler_path = 'scaler.pkl' 


model_streamlit = pickle.load(open(model_path, "rb"))
scaler_streamlit = pickle.load(open(scaler_path, "rb"))

# page config
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("🛃PredictiveChurn")

# Set the title and logo 
with st.sidebar:
    st.title("Customer Churn")
    st.image("https://static.vecteezy.com/system/resources/previews/031/971/395/non_2x/customer-churn-icon-design-vector.jpg",width=250,)


# user inputs (matching the simplified model's features)
senior_citizen = st.selectbox(
    "Senior Citizen?",
    options=[0, 1], # 0 for No, 1 for Yes
    format_func=lambda x: 'Yes' if x == 1 else 'No'
)

tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=72,
    value=1, 
    step=1
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=50.0,
    step=1.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=100.0,
    step=5.0
)

# Creating button
if st.button("Predict"):    

    input_data = pd.DataFrame({
        "SeniorCitizen": [senior_citizen],
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })


    scaled_input = scaler_streamlit.transform(input_data)
    scaled_input_df = pd.DataFrame(scaled_input, columns=input_data.columns)

    # Make prediction using model
    prediction = model_streamlit.predict(scaled_input_df)[0]
    prediction_proba = model_streamlit.predict_proba(scaled_input_df)[0]

    st.subheader("Prediction:")
    if prediction == 1:
        st.error(f"Customer will Churn ❌ (Probability: {prediction_proba[1]:.2f})")
    else:
        st.success(f"Customer will Stay ✅ (Probability: {prediction_proba[0]:.2f})")
