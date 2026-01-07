import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt

# -------------------------------------------------------
# Page Config
# -------------------------------------------------------
st.set_page_config(
    page_title="🏡 Real Estate Price Dashboard",
    layout="wide",
    page_icon="🏠"
)

# -------------------------------------------------------
# Material UI CSS
# -------------------------------------------------------
st.markdown("""
<style>

body {
    background-color: #F3F5F9;
}

/* Page container */
.block-container {
    padding-top: 1rem;
}

/* Material card style */
.card {
    background: white;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    transition: all 0.2s ease-in-out;
}

.card:hover {
    transform: translateY(-4px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.12);
}

/* Titles */
h2, h3, h4 {
    color: #1F2A44;
    font-weight: 700;
}

/* KPI labels */
.metric-label {
    color: #666;
    font-size: 13px;
    letter-spacing: .05rem;
}

/* Predicted price floating card */
.predict-box {
    background: linear-gradient(135deg,#1976D2,#42A5F5);
    color: white;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 15px 40px rgba(0,0,0,0.18);
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Load model & dataset
# -------------------------------------------------------
model = joblib.load("house_price_model.pkl")
df = pd.read_csv("house_data (1).csv")

# -------------------------------------------------------
# Title
# -------------------------------------------------------
st.markdown("<h1 style='text-align:center'>🏡 Real Estate Price Dashboard (PKR)</h1>", unsafe_allow_html=True)
st.write("")

# -------------------------------------------------------
# Layout: Inputs + Prediction
# -------------------------------------------------------
left, right = st.columns([2, 1])

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🎛 House Features")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        size = st.slider("Size (sq.ft)", 400, 6000, 1400)

    with c2:
        bedrooms = st.selectbox("Bedrooms", [1,2,3,4,5,6,7,8])

    with c3:
        bathrooms = st.selectbox("Bathrooms", [1,2,3,4,5,6])

    with c4:
        garage = st.slider("Garage Spaces", 0, 5, 1)

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    predicted_price = model.predict([[size, bedrooms, bathrooms, garage]])[0]

    st.markdown("<div class='predict-box'>", unsafe_allow_html=True)
    st.markdown("<h3>💰 Predicted Market Value</h3>", unsafe_allow_html=True)
    st.markdown(f"<h1>PKR {predicted_price:,.0f}</h1>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# -------------------------------------------------------
# KPI CARDS
# -------------------------------------------------------
st.subheader("📌 Market Snapshot")

k1, k2, k3 = st.columns(3)

with k1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='metric-label'>AVERAGE PRICE</div>", unsafe_allow_html=True)
    st.markdown(f"<h2>PKR {df.Price.mean():,.0f}</h2>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with k2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='metric-label'>HIGHEST PRICE</div>", unsafe_allow_html=True)
    st.markdown(f"<h2>PKR {df.Price.max():,.0f}</h2>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with k3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='metric-label'>LOWEST PRICE</div>", unsafe_allow_html=True)
    st.markdown(f"<h2>PKR {df.Price.min():,.0f}</h2>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------
# Charts in material cards
# -------------------------------------------------------
st.subheader("📊 Visual Insights")

a, b = st.columns(2)

with a:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    fig1 = px.scatter(df, x="Size", y="Price", color="Bedrooms",
                      title="Price vs Size (Colored by Bedrooms)")
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with b:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    fig2 = px.box(df, x="Bedrooms", y="Price",
                  title="Price Distribution by Bedrooms")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------
# Dataset Viewer
# -------------------------------------------------------
st.subheader("📄 Dataset")

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.dataframe(df, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)
