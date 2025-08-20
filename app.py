import streamlit as st
import pandas as pd
import joblib

# -----------------------
# Cache model & artifacts
# -----------------------
@st.cache_resource
def load_model_artifacts():
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_cols = joblib.load("models/feature_columns.pkl")
    return model, scaler, feature_cols

model, scaler, feature_cols = load_model_artifacts()

# -----------------------
# Page configuration
# -----------------------
st.set_page_config(page_title="Streaming Churn Predictor", page_icon="📺", layout="wide")

# -----------------------
# Title & header
# -----------------------
st.markdown("<h1 style='text-align:center; color:darkblue;'>Streaming Service Churn Predictor</h1>", unsafe_allow_html=True)
st.markdown("---")

# -----------------------
# Input section
# -----------------------
st.subheader("Subscriber Information")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 0, 100, 25)
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
    watch_hours = st.slider("Watch Hours (last month)", 0, 200, 10)
    last_login_days = st.slider("Days Since Last Login", 0, 365, 3)
    region = st.selectbox("Region", ["North", "South", "East", "West"])

with col2:
    device = st.selectbox("Device", ["Mobile", "Tablet", "TV", "Laptop"])
    monthly_fee = st.slider("Monthly Fee ($)", 0.0, 100.0, 9.99, step=0.01)
    payment_method = st.selectbox("Payment Method", ["Credit Card", "Debit Card", "Paypal"])
    number_of_profiles = st.slider("Number of Profiles", 1, 10, 1)
    avg_watch_time_per_day = st.slider("Avg Watch Time Per Day (hours)", 0.0, 24.0, 1.5, step=0.1)
    favorite_genre = st.selectbox("Favorite Genre", ["Action", "Drama", "Comedy", "Horror", "Sci-Fi"])

# -----------------------
# Prepare user input
# -----------------------
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
numeric_cols = ["age","watch_hours","last_login_days","monthly_fee","number_of_profiles","avg_watch_time_per_day"]
categorical_cols = ["gender", "subscription_type", "region", "device", "payment_method", "favorite_genre"]

# -----------------------
# Prediction button
# -----------------------
st.markdown("---")
if st.button("Predict Churn"):
    with st.spinner("Predicting churn..."):
        # One-hot encode & align columns
        input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
        input_encoded = input_encoded.reindex(columns=feature_cols, fill_value=0)
        # Scale numeric columns
        input_encoded[numeric_cols] = scaler.transform(input_encoded[numeric_cols])
        # Predict
        pred_class = model.predict(input_encoded)[0]
        pred_proba = model.predict_proba(input_encoded)[:,1][0]

        # Display result
        if pred_class == 1:
            st.markdown(f"""
                <div style='background-color:#ffcccc; padding:20px; border-radius:10px; text-align:center'>
                    <h2 style='color:red;'>⚠️ Likely to Churn</h2>
                    <p>Probability: <b>{pred_proba:.2f}</b></p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style='background-color:#ccffcc; padding:20px; border-radius:10px; text-align:center'>
                    <h2 style='color:green;'>✅ Likely to Stay</h2>
                    <p>Probability: <b>{pred_proba:.2f}</b></p>
                </div>
            """, unsafe_allow_html=True)
