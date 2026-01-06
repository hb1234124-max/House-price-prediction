import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

# -------------------------------
# Load model and dataset
# -------------------------------
model = joblib.load("house_price_model.pkl")
df = pd.read_csv("house_data.csv")

# -------------------------------
# Streamlit Page Setup
# -------------------------------
st.set_page_config(page_title="🏡 Real Estate Dashboard", layout="wide", page_icon="🏠")
st.markdown("<h1 style='text-align:center; color: darkblue;'>🏡 Real Estate Price Dashboard (PKR)</h1>", unsafe_allow_html=True)
st.write("Adjust the sliders to predict house prices dynamically!")

# -------------------------------
# Input sliders
# -------------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    size_input = st.slider("Size (sq.ft)", 100, 5000, 1000)
with col2:
    bedrooms_input = st.slider("Bedrooms", 1, 10, 3)
with col3:
    bathrooms_input = st.slider("Bathrooms", 1, 10, 2)
with col4:
    garage_input = st.slider("Garage Spaces", 0, 5, 1)

# -------------------------------
# Predict Price
# -------------------------------
predicted_price = model.predict([[size_input, bedrooms_input, bathrooms_input, garage_input]])[0]

st.markdown(f"""
<div style='background-color:#00BFFF;padding:20px;border-radius:15px'>
    <h2 style='text-align:center;color:white;'>💰 Predicted Price: PKR {predicted_price:,.0f}</h2>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Show Dataset
# -------------------------------
if st.checkbox("Show Dataset"):
    st.dataframe(df)

# -------------------------------
# 3D Visualization
# -------------------------------
st.subheader("3D Visualization: Size, Bedrooms, Price")
fig_3d = px.scatter_3d(df, x='Size', y='Bedrooms', z='Price',
                       color='Price', size='Garage', hover_data=['Bathrooms'],
                       title="Existing Houses: 3D View")
fig_3d.add_scatter3d(x=[size_input], y=[bedrooms_input], z=[predicted_price],
                     mode='markers', marker=dict(size=8, color='red'),
                     name='Predicted House')
st.plotly_chart(fig_3d, use_container_width=True)

# -------------------------------
# Predicted Price vs Size Chart
# -------------------------------
st.subheader("Predicted Price for Different Sizes")
sizes_range = np.linspace(500, 2500, 20)
pred_prices_range = model.predict(np.array([[s, bedrooms_input, bathrooms_input, garage_input] for s in sizes_range]))

fig, ax = plt.subplots()
ax.plot(sizes_range, pred_prices_range, color='green', marker='o')
ax.axhline(y=predicted_price, color='red', linestyle='--', label='Current Prediction')
ax.set_xlabel("Size (sq.ft)")
ax.set_ylabel("Predicted Price (PKR)")
ax.set_title("Predicted Price vs Size")
ax.legend()
st.pyplot(fig)

# -------------------------------
# Price Heatmap
# -------------------------------
st.subheader("Price Heatmap (Size vs Bedrooms)")
sizes = [600, 800, 1000, 1200, 1500, 1800, 2000, 2200, 2500]
bedrooms = [1, 2, 3, 4, 5]
heat_data = np.array([[model.predict([[s, b, bathrooms_input, garage_input]])[0] for b in bedrooms] for s in sizes])

fig_heat, ax = plt.subplots()
cax = ax.matshow(heat_data, cmap='YlOrRd')
fig_heat.colorbar(cax)
ax.set_xticks(range(len(bedrooms)))
ax.set_xticklabels(bedrooms)
ax.set_yticks(range(len(sizes)))
ax.set_yticklabels(sizes)
ax.set_xlabel("Bedrooms")
ax.set_ylabel("Size (sq.ft)")
ax.set_title("Price Heatmap (PKR)")
st.pyplot(fig_heat)

# -------------------------------
# Footer
# -------------------------------
st.markdown("<p style='text-align:center; color:gray;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)
