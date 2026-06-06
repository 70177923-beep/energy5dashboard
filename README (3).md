# ⚡ OWID Global Energy Dashboard

**SAP ID: 70177923**

An interactive energy data analysis and ML prediction dashboard built with Streamlit, using the Our World in Data energy dataset.

---

## 🔗 Data Source
[Our World in Data — Energy Dataset](https://github.com/owid/energy-data)

---

## 🚀 Features
- 🌍 Country & year range filters
- 📈 Energy trends (Renewable, Fossil, Solar, Wind, Hydro)
- 🔥 Feature correlation heatmap
- 🤖 Random Forest ML model to predict renewable energy share
- 📊 Feature importance visualization

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud

1. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
2. Click **New App**
3. Select this repository
4. Set **Main file**: `app.py`
5. Click **Deploy**

---

## 📁 Files

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit application |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |
