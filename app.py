# ---------- IMPORT ----------
import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Dashboard Analisis Penjualan",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    df = pd.read_csv("data/data_dummy_retail_store.csv")
    df["Tanggal_Pesanan"] = pd.to_datetime(df["Tanggal_Pesanan"], errors="coerce")
    df["Bulan"] = df["Tanggal_Pesanan"].dt.to_period("M").astype(str)
    return df

df_sales = load_data()

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    with open("models/model_sales.pkl", "rb") as f:
        model = pickle.load(f)          # hanya satu objek model
    return model

sales_model = load_model()

# ---------- SIDEBAR ----------
st.sidebar.header("Pengaturan & Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ("Overview Dashboard", "Prediksi Penjualan"))

# ===============================================================
# ==================  PAGE 1 : OVERVIEW DASHBOARD  ==============
# ===============================================================
if page == "Overview Dashboard":
    st.sidebar.markdown("### Filter Data Dashboard")

    # ----- Filter tanggal -----
    min_date = df_sales["Tanggal_Pesanan"].min().date()
    max_date = df_sales["Tanggal_Pesanan"].max().date()
    date_range = st.sidebar.date_input(
        "Pilih Rentang Tanggal:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    filtered_df = df_sales.copy()
    if len(date_range) == 2:
        start, end = map(pd.to_datetime, date_range)
        filtered_df = filtered_df[
            (filtered_df["Tanggal_Pesanan"] >= start)
            & (filtered_df["Tanggal_Pesanan"] <= end)
        ]

    # ----- Filter wilayah -----
    sel_regions = st.sidebar.multiselect(
        "Pilih Wilayah:",
        options=df_sales["Wilayah"].unique(),
        default=list(df_sales["Wilayah"].unique()),
    )
    filtered_df = filtered_df[filtered_df["Wilayah"].isin(sel_regions)]

    # ----- Filter kategori -----
    sel_cat = st.sidebar.multiselect(
        "Pilih Kategori Produk:",
        options=df_sales["Kategori"].unique(),
        default=list(df_sales["Kategori"].unique()),
    )
    filtered_df = filtered_df[filtered_df["Kategori"].isin(sel_cat)]

    # ---------- METRICS ----------
    st.subheader("Ringkasan Performa Penjualan")
    col1, col2, col3, col4 = st.columns(4)

    total_sales = filtered_df["Total_Penjualan"].sum()
    total_orders = filtered_df["OrderID"].nunique()
    avg_order_val = total_sales / total_orders if total_orders else 0
    total_products = filtered_df["Jumlah"].sum()

    col1.metric("Total Penjualan", f"Rp {total_sales:,.0f}")
    col2.metric("Jumlah Pesanan", f"{total_orders:,}")
    col3.metric("Avg. Order Value", f"Rp {avg_order_val:,.0f}")
    col4.metric("Produk Terjual", f"{total_products:,}")

    # ---------- TREN PENJUALAN BULANAN ----------
    st.subheader("Tren Penjualan Bulanan")
    sales_by_month = (
        filtered_df.groupby("Bulan")["Total_Penjualan"].sum().reset_index()
    )
    sales_by_month["Bulan"] = (
        pd.to_datetime(sales_by_month["Bulan"]).dt.to_period("M").astype(str)
    )
    sales_by_month = sales_by_month.sort_values("Bulan")

    fig_monthly = px.line(
        sales_by_month,
        x="Bulan",
        y="Total_Penjualan",
        title="Total Penjualan per Bulan",
        markers=True,
        height=400,
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

    # ---------- TOP PRODUCT & PIE DISTRIBUSI ----------
    st.subheader("Top Product & Distribusi Penjualan")
    col_vis1, col_vis2 = st.columns(2)

    # -- kolom 1: Top 10 produk --
    with col_vis1:
        st.write("#### Top 10 Produk")
        top_products = (
            filtered_df.groupby("Produk")["Total_Penjualan"]
            .sum()
            .nlargest(10)
            .reset_index()
        )
        fig_top = px.bar(
            top_products,
            x="Total_Penjualan",
            y="Produk",
            orientation="h",
            title="Top 10 Produk Berdasarkan Total Penjualan",
            color="Total_Penjualan",
            color_continuous_scale=px.colors.sequential.Plasma[::-1],
            height=400,
        )
        fig_top.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_top, use_container_width=True)

    # -- kolom 2: Pie distribusi kategori --
    with col_vis2:
        st.write("#### Distribusi Penjualan per Kategori")
        sales_by_cat = (
            filtered_df.groupby("Kategori")["Total_Penjualan"].sum().reset_index()
        )
        fig_pie = px.pie(
            sales_by_cat,
            values="Total_Penjualan",
            names="Kategori",
            title="Proporsi Penjualan per Kategori",
            hole=0.3,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# ===============================================================
# =================  PAGE 2 : PREDIKSI PENJUALAN  ===============
# ===============================================================
else:
    st.subheader("Prediksi Penjualan per Baris Data")
    df_pred = df_sales.copy()
    df_pred["day_ordinal"] = df_pred["Tanggal_Pesanan"].dt.day

    # Jika model memiliki feature_names_in_
    if hasattr(sales_model, "feature_names_in_"):
        X = df_pred[sales_model.feature_names_in_]
    else:
        # contoh fitur numerik minimal
        X = df_pred[["Bulan", "Diskon", "Harga_Satuan", "Jumlah", "day_ordinal"]]

    df_pred["Prediksi"] = sales_model.predict(X)

    st.dataframe(
        df_pred[["OrderID", "Prediksi"]].head(15), use_container_width=True
    )

    st.markdown("### Distribusi Nilai Prediksi")
    fig_hist = px.histogram(df_pred, x="Prediksi", nbins=30)
    st.plotly_chart(fig_hist, use_container_width=True)