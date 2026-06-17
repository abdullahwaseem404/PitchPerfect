import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor

st.set_page_config(page_title="PitchPerfect FIFA Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("fifa_data.csv")

    drop_cols = ["Unnamed: 0", "ID", "Photo", "Flag", "Club Logo"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df

fifa = load_data()

def height_to_cm(h):
    try:
        feet, inches = h.split("'")
        return int(feet) * 30.48 + int(inches) * 2.54
    except:
        return np.nan

def weight_to_kg(w):
    try:
        return int(w.replace("lbs", "")) * 0.453592
    except:
        return np.nan

def money_to_number(v):
    if pd.isna(v):
        return np.nan
    v = str(v).replace("€", "")
    if "M" in v:
        return float(v.replace("M", "")) * 1_000_000
    if "K" in v:
        return float(v.replace("K", "")) * 1_000
    return np.nan

fifa["Height_cm"] = fifa["Height"].astype(str).apply(height_to_cm)
fifa["Weight_kg"] = fifa["Weight"].astype(str).apply(weight_to_kg)
fifa["Wage_EUR"] = fifa["Wage"].apply(money_to_number)
fifa["Value_EUR"] = fifa["Value"].apply(money_to_number)

st.sidebar.header("Filters")

nationalities = st.sidebar.multiselect(
    "Nationality", sorted(fifa["Nationality"].dropna().unique())
)

clubs = st.sidebar.multiselect(
    "Club", sorted(fifa["Club"].dropna().unique())
)

filtered = fifa.copy()

if nationalities:
    filtered = filtered[filtered["Nationality"].isin(nationalities)]

if clubs:
    filtered = filtered[filtered["Club"].isin(clubs)]

st.title("⚽ PitchPerfect – FIFA Player Data Dashboard")
st.subheader("Interactive Analytics + Machine Learning Insights")

st.dataframe(filtered.head())

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🌍 Top Nationalities")
    counts = filtered["Nationality"].value_counts().head(5)
    fig, ax = plt.subplots()
    ax.pie(counts, labels=counts.index, autopct="%1.1f%%")
    st.pyplot(fig)

with col2:
    st.markdown("### 🎂 Age Distribution")
    fig, ax = plt.subplots()
    ax.hist(filtered["Age"], bins=20)
    st.pyplot(fig)

with col3:
    st.markdown("### 🦶 Preferred Foot")
    fig, ax = plt.subplots()
    filtered["Preferred Foot"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
    st.pyplot(fig)

st.header("🤖 Machine Learning Analysis")

features = [
    "Age", "Height_cm", "Weight_kg",
    "Acceleration", "SprintSpeed", "Stamina", "Strength"
]

target = "Overall"

ml_df = filtered[features + [target]].dropna()

if len(ml_df) > 50:

    X = ml_df[features]
    y = ml_df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    rf_r2 = r2_score(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))


    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)

    xgb_r2 = r2_score(y_test, xgb_pred)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))


    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌲 Random Forest")
        st.write(f"R² Score: {rf_r2:.3f}")
        st.write(f"RMSE: {rf_rmse:.2f}")

    with col2:
        st.subheader("⚡ XGBoost")
        st.write(f"R² Score: {xgb_r2:.3f}")
        st.write(f"RMSE: {xgb_rmse:.2f}")


    st.subheader("📊 Feature Importance (Random Forest)")

    importances = pd.Series(rf.feature_importances_, index=features)
    fig, ax = plt.subplots()
    importances.sort_values().plot(kind="barh", ax=ax)
    st.pyplot(fig)

else:
    st.warning("Not enough data for ML after filtering.")


st.header("🎯 Player Rating Predictor")

age = st.slider("Age", 16, 45, 25)
height = st.slider("Height (cm)", 150, 210, 175)
weight = st.slider("Weight (kg)", 50, 110, 70)
acc = st.slider("Acceleration", 20, 100, 60)
speed = st.slider("Sprint Speed", 20, 100, 60)
stamina = st.slider("Stamina", 20, 100, 60)
strength = st.slider("Strength", 20, 100, 60)

if st.button("Predict Overall Rating"):

    input_data = np.array([[age, height, weight, acc, speed, stamina, strength]])

    prediction = rf.predict(input_data)[0]

    st.success(f"Predicted Overall Rating: {prediction:.2f}")