import streamlit as st
import pandas as pd
import requests
import os
import matplotlib.pyplot as plt

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="House Price Prediction", layout="centered")
st.title("Ames House Price Prediction")

# -------------------------------------------------
# Health check
# -------------------------------------------------
st.subheader("API Status")

try:
    r = requests.get(f"{API_URL}/health", timeout=2)
    if r.status_code == 200:
        st.success("API is running")
    else:
        st.error("API responded but not healthy")
except Exception as e:
    st.error(f"API not reachable: {e}")
    st.stop()

# =================================================
# SECTION 1 — BATCH PREDICTION
# =================================================
st.divider()
st.subheader("Batch Prediction (Unlabeled CSV)")

uploaded_file = st.file_uploader(
    "Upload CSV for prediction",
    type=["csv"],
    key="predict_csv"
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    if st.button("Run Prediction"):
        with st.spinner("Predicting..."):
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "text/csv"
                )
            }

            try:
                response = requests.post(
                    f"{API_URL}/predict-csv",
                    files=files,
                    timeout=60
                )

                if response.status_code != 200:
                    st.error(f"API error: {response.text}")
                    st.stop()

                result = response.json()

            except Exception as e:
                st.error(f"Request failed: {e}")
                st.stop()

        preds = result["predictions"]

        st.success(f"Predictions completed for {result['n_rows']} rows")

        df_results = df.copy()
        df_results["PredictedPrice"] = preds

        st.subheader("Predictions")
        st.dataframe(df_results.head())

        # Visualization
        st.subheader("Prediction Distribution")
        fig, ax = plt.subplots()
        ax.hist(preds, bins=30)
        ax.set_xlabel("Predicted House Price")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        csv_out = df_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download predictions as CSV",
            csv_out,
            file_name="predictions.csv",
            mime="text/csv"
        )

# =================================================
# SECTION 2 — LABELED DATA UPLOAD (RETRAINING)
# =================================================
st.divider()
st.subheader("Upload Labeled Data (Check for Retraining)")

st.info(
    "This data will be stored and checked for drift by the monitoring service."
)

labeled_file = st.file_uploader(
    "Upload labeled CSV",
    type=["csv"],
    key="labeled_csv"
)

if labeled_file is not None:
    df_labeled = pd.read_csv(labeled_file)
    st.write("Preview of labeled data:")
    st.dataframe(df_labeled.head())

    if st.button("Upload for Drift Monitoring"):
        with st.spinner("Uploading labeled data..."):
            files = {
                "file": (
                    labeled_file.name,
                    labeled_file.getvalue(),
                    "text/csv"
                )
            }

            try:
                response = requests.post(
                    f"{API_URL}/upload-labeled-data",
                    files=files,
                    timeout=30
                )

                if response.status_code != 200:
                    st.error(f"API error: {response.text}")
                    st.stop()

                result = response.json()

            except Exception as e:
                st.error(f"Upload failed: {e}")
                st.stop()

        st.success("Labeled data uploaded successfully")
        st.json(result)
