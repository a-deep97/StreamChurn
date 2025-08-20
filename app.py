import streamlit as st
import pandas as pd
import joblib

# Load artifacts
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_cols = joblib.load("models/feature_columns.pkl")

st.title("Video Streaming Churn Prediction")

# User Inputs
st.header("Subscriber Information")

# Collect input
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", 0, 120, 25)
subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
watch_hours = st.number_input("Watch Hours in last month", 0.0, 200.0, 10.0)
last_login_days = st.number_input("Days since last login", 0, 365, 3)
region = st.selectbox("Region", ["North", "South", "East", "West"])
device = st.selectbox("Device", ["Mobile", "Tablet", "TV", "Laptop"])
monthly_fee = st.number_input("Monthly Fee", 0.0, 100.0, 9.99)
payment_method = st.selectbox("Payment Method", ["Credit Card", "Debit Card", "Paypal"])
number_of_profiles = st.number_input("Number of Profiles", 1, 10, 1)
avg_watch_time_per_day = st.number_input("Average Watch Time Per Day (hours)", 0.0, 24.0, 1.5)
favorite_genre = st.selectbox("Favorite Genre", ["Action", "Drama", "Comedy", "Horror", "Sci-Fi"])

# Create DataFrame
user_input = {
    "age": age,
    "gender": gender,
    "subscription_type": subscription_type,
    "watch_hours": watch_hours,
    "last_login_days": last_login_days,
    "region": region,
    "device": device,
    "monthly_fee": monthly_fee,
    "payment_method": payment_method,
    "number_of_profiles": number_of_profiles,
    "avg_watch_time_per_day": avg_watch_time_per_day,
    "favorite_genre": favorite_genre
}

input_df = pd.DataFrame([user_input])

# Preprocess
numeric_cols = ["age","watch_hours","last_login_days","monthly_fee","number_of_profiles","avg_watch_time_per_day"]
categorical_cols = ["gender", "subscription_type", "region", "device", "payment_method", "favorite_genre"]

# One-hot encode
input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

# Align columns with training
input_encoded = input_encoded.reindex(columns=feature_cols, fill_value=0)

# Scale numeric columns
input_encoded[numeric_cols] = scaler.transform(input_encoded[numeric_cols])

# Predict on button click
if st.button("Predict Churn"):
    pred_proba = model.predict_proba(input_encoded)[:,1][0]
    pred_class = model.predict(input_encoded)[0]

    if pred_class == 1:
        st.warning(f"⚠️ Likely to churn. Probability: {pred_proba:.2f}")
    else:
        st.success(f"✅ Likely to stay. Probability: {pred_proba:.2f}")
