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
    page_title="Real Estate Price Dashboard",
    layout="wide",
    page_icon="🏠"
)

# -------------------------------------------------------
# Animated Gradient Background + Modern UI
# -------------------------------------------------------
st.markdown("""
<style>

:root {
    --primary:#6C63FF;
    --secondary:#FF6584;
    --accent:#00C9A7;
}

body {
    background: linear-gradient(120deg,#6C63FF,#00C9A7,#FF6584);
    background-size:400% 400%;
    animation: gradientMove 10s infinite;
}

@keyframes gradientMove {
    0% {background-position:0% 50%;}
    50% {background-position:100% 50%;}
    100% {background-position:0% 50%;}
}

/* Glass effect cards */
.card {
    background: rgba(255,255,255,0.18);
    padding: 25px;
    border-radius: 18px;
    backdrop-filter: blur(10px);
    box-shadow: 0 10px 25px rgba(0,0,0,0.12);
    transition: .2s ease-in-out;
}

.card:hover{
    transform:translateY(-5px);
}

/* Prediction box */
.predict-box {
    background: linear-gradient(135deg,#FF6584,#6C63FF);
    color:white;
    padding:30px;
    border-radius:20px;
    box-shadow: 0 10px 30px rgba(0,0,0,.2);
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Load model and data
# -------------------------------------------------------
model = joblib.load("house_price_model.pkl")
df = pd.read_csv("house_data.csv")

# Add useful engineered metric
df["Price_per_sqft"] = df["Price"] / df["Size"]

# -------------------------------------------------------
# Sidebar Navigation
# -------------------------------------------------------
st.sidebar.title("🔎 Navigation")
page = st.sidebar.selectbox("Go to", ["Dashboard", "Dataset", "About App"])

# -------------------------------------------------------
# MAIN DASHBOARD
# -------------------------------------------------------
if page == "Dashboard":

    st.markdown("<h1 style='text-align:center;color:white'>🏡 Real Estate Price Dashboard (PKR)</h1>",
                unsafe_allow_html=True)

    left, right = st.columns([2,1])

    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🎛 Select House Features")

        c1,c2,c3,c4 = st.columns(4)

        with c1:
            size = st.slider("Size (sq.ft)", 400,6000,1400)

        with c2:
            bedrooms = st.selectbox("Bedrooms", [1,2,3,4,5,6,7,8])

        with c3:
            bathrooms = st.selectbox("Bathrooms", [1,2,3,4,5,6])

        with c4:
            garage = st.slider("Garage Capacity", 0,5,1)

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        predicted_price = model.predict([[size,bedrooms,bathrooms,garage]])[0]

        st.markdown("<div class='predict-box'>", unsafe_allow_html=True)
        st.subheader("💰 Predicted Market Price")

        st.markdown(f"<h1>PKR {predicted_price:,.0f}</h1>", unsafe_allow_html=True)

        st.success("Prediction generated successfully ✔")
        st.balloons()
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    # KPI cards
    st.subheader("📌 Market Snapshot")
    k1,k2,k3,k4 = st.columns(4)

    with k1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.metric("Average Price", f"PKR {df.Price.mean():,.0f}")
        st.markdown("</div>", unsafe_allow_html=True)

    with k2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.metric("Highest Price", f"PKR {df.Price.max():,.0f}")
        st.markdown("</div>", unsafe_allow_html=True)

    with k3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.metric("Lowest Price", f"PKR {df.Price.min():,.0f}")
        st.markdown("</div>", unsafe_allow_html=True)

    with k4:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.metric("Avg Price per Sqft", f"PKR {df.Price_per_sqft.mean():,.0f}")
        st.markdown("</div>", unsafe_allow_html=True)

    # Charts
    st.subheader("📊 Interactive Visual Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        df = df.dropna(subset=["Size", "Price", "Bedrooms"])
        fig = px.scatter(df, x="Size", y="Price",
                         color="Bedrooms",
                         trendline="ols",
                         title="📈 Price vs Size")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        fig2 = px.density_heatmap(df, x="Bedrooms", y="Bathrooms", z="Price",
                                  title="🔥 Price Heatmap")
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------
# DATASET PAGE
# -------------------------------------------------------
elif page == "Dataset":
    st.subheader("📄 Dataset Viewer")
    st.dataframe(df, use_container_width=True)

# -------------------------------------------------------
# ABOUT PAGE
# -------------------------------------------------------
else:
    st.title("ℹ️ About this Dashboard")
    st.write("""
    This colorful dashboard helps predict real-estate prices using machine learning.  
    You can explore data trends, price patterns, and generate predictions interactively.
    """)

