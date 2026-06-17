# ⚽ PitchPerfect – FIFA Player Data Dashboard

An interactive **FIFA player analytics dashboard** built with Streamlit, featuring data visualization, filtering, and machine learning-based player rating prediction.

---

## 🚀 Features

* 📊 Interactive data filtering (Nationality & Club)
* 📈 Visual analytics:
  * Top nationalities
  * Age distribution
  * Preferred foot breakdown
* 🤖 Machine learning models:
  * Random Forest Regressor
  * XGBoost Regressor
* 📐 Feature importance visualization
* 🎯 Player rating prediction tool
* ⚡ Fast and responsive Streamlit UI

---

## 🧠 Machine Learning Models
### 🌲 Random Forest
* Ensemble-based model
* Handles non-linearity well
* Provides feature importance
### ⚡ XGBoost
* Gradient boosting model
* High performance and accuracy
* Efficient with structured data

---

## ⚙️ Installation

```bash id="fifa13"
git clone https://github.com/abdullahwaseem404/PitchPerfect.git
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the Streamlit app:

```bash id="fifa15"
streamlit run app.py
```

---

## 🎛️ Dashboard Controls

* **Nationality Filter** → Filter players by country
* **Club Filter** → Filter players by club
* Dynamic charts update in real-time

---

## 📊 Visualizations

* 🌍 Top 5 Nationalities (Pie Chart)
* 🎂 Age Distribution (Histogram)
* 🦶 Preferred Foot Distribution (Pie Chart)

---

## 🤖 ML Evaluation Metrics

* **R² Score** → Model accuracy
* **RMSE** → Prediction error

---

## 📐 Feature Engineering

* Height converted to **cm**
* Weight converted to **kg**
* Wage & Value converted to numeric (€ → EUR)

---

## 🎯 Player Rating Predictor

Users can input:

* Age
* Height
* Weight
* Acceleration
* Sprint Speed
* Stamina
* Strength

👉 Model predicts the **Overall FIFA Rating**

---
