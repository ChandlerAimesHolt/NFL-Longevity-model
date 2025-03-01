import numpy as np
import streamlit as st
import pickle
import requests
from sklearn.preprocessing import StandardScaler

# ✅ Download & Load Pre-Trained Model
model_url = "https://github.com/ChandlerAimesHolt/NFL-Longevity-model/raw/refs/heads/main/classifier.pkl"

try:
    response = requests.get(model_url)
    if response.status_code == 200:
        with open("classifier.pkl", "wb") as file:
            file.write(response.content)

        # Load the model
        with open("classifier.pkl", "rb") as model_file:
            model = pickle.load(model_file)
    else:
        st.error("Failed to download the model. Please check the URL.")
        st.stop()

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ✅ Streamlit UI
st.title("Defensive Lineman Longevity Predictor")
st.write("Enter player attributes to predict if a defensive lineman will play at least 8 years in the NFL.")

# 🔹 User Inputs (Sliders and Radio Buttons)
pos = st.radio("Position", options=[0, 1], format_func=lambda x: "Defensive End (1)" if x == 1 else "Defensive Tackle (0)")
ras = st.slider("Relative Athletic Score (RAS)", min_value=0.0, max_value=10.0, step=0.1)
yr2_av = st.slider("2nd Year AV", min_value=0.0, max_value=6.0, step=1.0)
yr3_av = st.slider("3rd Year AV", min_value=0.0, max_value=6.0, step=1.0)
yr4_av = st.slider("4th Year AV", min_value=0.0, max_value=6.0, step=1.0)
age = st.slider("Age", min_value=21, max_value=24, step=1)

# 🔹 Prediction Button
if st.button("Predict"):
    # ✅ Prepare input features
    features = np.array([[pos, ras, yr2_av, yr3_av, yr4_av, age]])

    # ✅ Scale features (IMPORTANT: This must match training scaling)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)  # Fit & transform inputs

    # ✅ Make prediction
    prediction = model.predict(features_scaled)
    prediction_proba = model.predict_proba(features_scaled)

    # ✅ Display Results
    result_text = "Yes ✅" if prediction[0] == 1 else "No ❌"
    st.success(f"Will this player play at least 8 years?: {result_text}")

    # 🔍 Debugging: Show Prediction Probabilities
    st.write("🔍 Prediction Probabilities:", prediction_proba)
