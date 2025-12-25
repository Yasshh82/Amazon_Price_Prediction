import streamlit as st
import requests

st.set_page_config(page_title="Amazon Price Predictor", page_icon="📦")

st.title("📦 Amazon Product Price Predictor")
st.markdown("Enter product details below to get an AI-powered price estimate.")

with st.form("price_form"):
    desc = st.text_area("Product Description", placeholder="Item Name: La Victoria Sauce...")
    col1, col2 = st.columns(2)
    with col1:
        ps = st.number_input("Pack Size", min_value=1, value=1)
    with col2:
        tm = st.number_input("Total Measure", min_value=0.1, value=10.0)

    submit = st.form_submit_button("Estimate Price")

if submit:
    if desc:
        payload = {
            "description": desc,
            "pack_size": ps,
            "total_measure": tm}
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)

        if response.status_code == 200:
            result = response.json()
            st.success(f"### Estimated Price: ${result['estimated_price']}")
            st.info(f"Brand Detected: {result['brand_detected'].title()}")
        else:
            st.error("Error connecting to the backend")
    else:
        st.warning("Please enter a description.")