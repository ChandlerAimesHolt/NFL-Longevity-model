import pandas as pd
import numpy as np
import streamlit as st
import pickle
import requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ✅ Fix dataset URL (Use correct GitHub raw link)
data_url = "https://raw.githubusercontent.com/ChandlerAimesHolt/NFL-Longevity-model/main/Dlineman%20prediction.csv"

try:
    Final_Data = pd.read_csv(data_url)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# ✅ Prepare features and target
X = Final_Data.drop(columns=['Name', 'career_length', '8_or_more'])
y = Final_Data['8_or_more']

# ✅ Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ Train Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)

# ✅ Download model from GitHub
model_url = "https://raw.githubusercontent.com/ChandlerAimesHolt/NFL-Longevity-model/main/classifier.pkl"

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
    features = np.array([[pos, ras, yr2_av, yr3_av, yr4_av, age]])
    prediction = model.predict(features)

    # ✅ Display Result
    result_text = "Yes ✅" if prediction[0] == 1 else "No ❌"
    st.success(f"Will this player play at least 8 years?: {result_text}")
