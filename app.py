import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="FIFA Dashboard", layout="wide")

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

fifa["Height_cm"] = fifa["Height"].astype(str).apply(height_to_cm)
fifa["Weight_kg"] = fifa["Weight"].astype(str).apply(weight_to_kg)


def money_to_number(v):
    if pd.isna(v):
        return np.nan
    v = str(v).replace("€", "")
    if "M" in v:
        return float(v.replace("M", "")) * 1_000_000
    if "K" in v:
        return float(v.replace("K", "")) * 1_000
    return np.nan

fifa["Wage_EUR"] = fifa["Wage"].apply(money_to_number)
fifa["Value_EUR"] = fifa["Value"].apply(money_to_number)


st.sidebar.header("Filters")

nationalities = st.sidebar.multiselect(
    "Select Nationality",
    sorted(fifa["Nationality"].dropna().unique())
)

clubs = st.sidebar.multiselect(
    "Select Club",
    sorted(fifa["Club"].dropna().unique())
)

filtered = fifa.copy()

if nationalities:
    filtered = filtered[filtered["Nationality"].isin(nationalities)]

if clubs:
    filtered = filtered[filtered["Club"].isin(clubs)]


st.title("⚽ FIFA Player Data Visualization Dashboard")
st.subheader("📄 Dataset Preview")
st.dataframe(filtered.head())


col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🌍 Top Nationalities")
    counts = filtered["Nationality"].value_counts().head(5)
    fig, ax = plt.subplots()
    ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=140)
    ax.axis("equal")
    st.pyplot(fig)

with col2:
    st.markdown("### 🎂 Age Distribution")
    fig, ax = plt.subplots()
    ax.hist(filtered["Age"], bins=20, edgecolor="black")
    ax.set_xlabel("Age")
    ax.set_ylabel("Players")
    st.pyplot(fig)

with col3:
    st.markdown("### 🦶 Preferred Foot")
    fig, ax = plt.subplots()
    filtered["Preferred Foot"].value_counts().plot(
        kind="pie", autopct="%1.1f%%", ax=ax
    )
    ax.set_ylabel("")
    st.pyplot(fig)

col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("### 🏆 Top Clubs by Rating")
    club_rating = (
        filtered.groupby("Club")["Overall"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    fig, ax = plt.subplots()
    club_rating.plot(kind="bar", ax=ax)
    ax.set_ylabel("Avg Overall")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    st.pyplot(fig)

with col5:
    st.markdown("### 📈 Overall vs Potential")
    fig, ax = plt.subplots()
    ax.scatter(filtered["Overall"], filtered["Potential"], alpha=0.5)
    ax.set_xlabel("Overall")
    ax.set_ylabel("Potential")
    ax.set_xticks(range(40, 100, 5))
    ax.set_yticks(range(40, 100, 5))
    st.pyplot(fig)

with col6:
    st.markdown("### 📍 Positions")
    fig, ax = plt.subplots()
    filtered["Position"].value_counts().head(10).plot(kind="bar", ax=ax)
    ax.set_ylabel("Players")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

col7, col8, col9 = st.columns(3)

with col7:
    st.markdown("### 💰 Wage vs Overall")
    clean_wage = filtered.dropna(subset=["Wage_EUR"])
    fig, ax = plt.subplots()
    ax.scatter(clean_wage["Overall"], clean_wage["Wage_EUR"] / 1000, alpha=0.4)
    ax.set_xlabel("Overall Rating")
    ax.set_ylabel("Wage (Thousand €)")
    ax.set_xticks(range(40, 100, 5))
    st.pyplot(fig)

with col8:
    st.markdown("### 📏 Height vs Weight")
    clean_hw = filtered.dropna(subset=["Height_cm", "Weight_kg"])
    fig, ax = plt.subplots()
    ax.scatter(clean_hw["Height_cm"], clean_hw["Weight_kg"], alpha=0.4)
    ax.set_xlabel("Height (cm)")
    ax.set_ylabel("Weight (kg)")
    st.pyplot(fig)

with col9:
    st.markdown("### 🔥 Attribute Correlation")
    attrs = ["Overall", "Potential", "Acceleration", "SprintSpeed", "Strength", "Stamina"]
    corr = filtered[attrs].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)