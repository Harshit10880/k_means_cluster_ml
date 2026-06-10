import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("kmeans_model.joblib")

# Load dataset
df = pd.read_csv("iris_data.csv")

st.set_page_config(
    page_title="Iris KMeans Clustering",
    page_icon="🌸",
    layout="wide"
)

st.title("🌸 Iris Flower Clustering Dashboard")

st.sidebar.header("Flower Measurements")

sepal_length = st.sidebar.slider(
    "Sepal Length",
    4.0,
    8.0,
    5.1
)

sepal_width = st.sidebar.slider(
    "Sepal Width",
    2.0,
    5.0,
    3.5
)

input_df = pd.DataFrame({
    "sepal_length": [sepal_length],
    "sepal_width": [sepal_width]
})

cluster = model.predict(input_df)[0]

st.subheader("Prediction")

st.success(f"Assigned Cluster : {cluster}")

# Dataset Clusters
X = df[["sepal_length", "sepal_width"]]

df["Cluster"] = model.predict(X)

# Dynamic Chart
fig, ax = plt.subplots(figsize=(8, 6))

for c in sorted(df["Cluster"].unique()):

    cluster_data = df[df["Cluster"] == c]

    ax.scatter(
        cluster_data["sepal_length"],
        cluster_data["sepal_width"],
        label=f"Cluster {c}"
    )

# Cluster Centers
centers = model.cluster_centers_

ax.scatter(
    centers[:, 0],
    centers[:, 1],
    marker="X",
    s=250,
    label="Centroids"
)

# User Point
ax.scatter(
    sepal_length,
    sepal_width,
    marker="*",
    s=400,
    label="Your Flower"
)

ax.set_xlabel("Sepal Length")
ax.set_ylabel("Sepal Width")
ax.set_title("KMeans Clustering Visualization")

ax.legend()

st.pyplot(fig)

# Cluster Statistics
st.subheader("Cluster Distribution")

cluster_counts = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)

st.bar_chart(cluster_counts)