import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(page_title="Units Sold Prediction", layout="centered")

st.title("🛒 Units Sold Prediction – Use Case 2")
st.write("Enter values to predict actual units sold")

# --------------------------------
# SIMULATED TRAINING DATA
# --------------------------------
np.random.seed(42)

train_df = pd.DataFrame({
    "Price": np.random.uniform(50, 500, 300),
    "Discount": np.random.uniform(0, 0.5, 300),
    "Inventory_Level": np.random.randint(50, 1000, 300),
    "Promotion": np.random.randint(0, 2, 300),
    "Weather": np.random.randint(0, 3, 300),   # 0=Normal,1=Rainy,2=Extreme
    "Seasonality": np.random.randint(0, 2, 300)
})

train_df["Units_Sold"] = (
    0.6 * train_df["Inventory_Level"]
    - 0.5 * train_df["Price"]
    + 200 * train_df["Discount"]
    + 150 * train_df["Promotion"]
    - 50 * train_df["Weather"]
    + 120 * train_df["Seasonality"]
)

X = train_df.drop("Units_Sold", axis=1)
y = train_df["Units_Sold"]

# --------------------------------
# MODEL TRAINING
# --------------------------------
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)
model.fit(X, y)

# --------------------------------
# USER INPUTS
# --------------------------------
st.sidebar.header("Input Features")

price = st.sidebar.number_input("Price", min_value=1.0, value=100.0)
discount = st.sidebar.slider("Discount (0–0.5)", 0.0, 0.5, 0.1)
inventory = st.sidebar.number_input("Inventory Level", min_value=0, value=500)
promotion = st.sidebar.selectbox("Promotion (0 = No, 1 = Yes)", [0, 1])

weather_label = st.sidebar.selectbox(
    "Weather Condition", ["Normal", "Rainy", "Extreme"]
)
weather_map = {"Normal": 0, "Rainy": 1, "Extreme": 2}

seasonality = st.sidebar.selectbox("Seasonality (0 = No, 1 = Yes)", [0, 1])

input_df = pd.DataFrame({
    "Price": [price],
    "Discount": [discount],
    "Inventory_Level": [inventory],
    "Promotion": [promotion],
    "Weather": [weather_map[weather_label]],
    "Seasonality": [seasonality]
})

st.subheader("User Input")
st.write(input_df)

# --------------------------------
# PREDICTION
# --------------------------------
if st.button("Predict Units Sold"):
    prediction = model.predict(input_df)[0]
    st.success(f"📦 Predicted Units Sold: {int(prediction)}")
