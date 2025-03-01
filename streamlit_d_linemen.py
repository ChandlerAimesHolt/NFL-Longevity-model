import pandas as pd
url = 'https://raw.githubusercontent.com/ChandlerAimesHolt/NFL-Longevity-model/refs/heads/main/Dlineman%20prediction'
Final_Data = pd.read_csv(url)

from sklearn.model_selection import train_test_split
X = Final_Data.drop(columns = ['Name', 'career_length', '8_or_more'])
y = Final_Data['8_or_more']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
log_reg = LogisticRegression()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

log_reg.fit(X_train_scaled, y_train)

import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open(r"https://github.com/ChandlerAimesHolt/NFL-Longevity-model/raw/refs/heads/main/classifier.pkl", "rb") as model_file:
    model = pickle.load(model_file)


st.title("Defensive Lineman Longevity")
st.write("Enter player attributes to predict performance.")

# Input sliders for features
pos = st.radio("Position (DE=1, DT=0)", options=[0, 1])
ras = st.slider("Relative Athletic Score (RAS)", min_value=0.0, max_value=10.0, step=0.1)
yr2_av = st.slider("2nd Year AV", min_value=0.0, max_value=6.0, step=1.0)
yr3_av = st.slider("3rd Year AV", min_value=0.0, max_value=6.0, step=1.0)
yr4_av = st.slider("4th Year AV", min_value=0.0, max_value=6.0, step=1.0)
age = st.slider("Age", min_value=21, max_value=24, step=1)

if st.button("Predict"):
    # Prepare features
    features = np.array([[pos, ras, yr2_av, yr3_av, yr4_av, age]])
    prediction = model.predict(features)
    st.write(f"Will this player play at least 8 years?: {prediction[0]}")
