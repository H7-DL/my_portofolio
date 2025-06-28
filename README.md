# 📊 Streamlit Retail Dashboard

This project is an interactive dashboard built with **Streamlit** to analyze and predict retail sales performance based on historical transaction data.

## 🔍 Features

- ✅ KPI summary cards (Total Sales, Orders, AOV, Products Sold)
- 📈 Monthly sales trend analysis (line chart)
- 🔝 Top 10 best-selling products (bar chart)
- 🧁 Sales distribution by category (donut pie chart)
- 🤖 Predictive model for daily sales using a trained ML model (`.pkl`)

---

## 📁 Folder Structure
my_portofolio/
│
├── data/
│ └── data_dummy_retail_store.csv # Dataset (ignored by git)
│
├── models/
│ └── model_sales.pkl # Trained ML model (ignored by git)
│
├── app.py # Streamlit app main script
├── .gitignore # Git ignore file
└── README.md # Project documentation
---

## 🚀 How to Run

1. **Install dependencies:**

   ```bash
   pip install streamlit pandas plotly

## 📊 Dashboard Preview

![Dashboard Preview](screenshots/SS%20Dashboard.png)
