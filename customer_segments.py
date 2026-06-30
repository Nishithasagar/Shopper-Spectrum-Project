# Customer Segementation

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load cleaned data
df = pd.read_csv(
    "cleaned_online_retail.csv",
    parse_dates=["InvoiceDate"]
)

# Create RFM table
reference_date = df["InvoiceDate"].max()

rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x:
        (reference_date - x.max()).days,

    "InvoiceNo": "count",

    "TotalAmount": "sum"
})

rfm.columns = [
    "Recency",
    "Frequency",
    "Monetary"
]

# Scale values
scaler = StandardScaler()

rfm_scaled = scaler.fit_transform(
    rfm
)

print(rfm.head())
print(rfm_scaled.shape)

from sklearn.cluster import KMeans

# Train final model
kmeans = KMeans(
    n_clusters=5,
    random_state=42
)

rfm["Cluster"] = kmeans.fit_predict(
    rfm_scaled
)

print(
    rfm["Cluster"]
    .value_counts()
)

# Cluster Summary

cluster_summary = (
    rfm.groupby("Cluster")
    [["Recency","Frequency","Monetary"]]
    .mean()
)

print(cluster_summary)

# Assign labels

labels = {
    0:"High-Value",
    1:"Regular",
    2:"Occasional",
    3:"At-Risk",
    4:"New Customer"
}

rfm["Segment"] = (
    rfm["Cluster"]
    .map(labels)
)

print(
    rfm[
        ["Cluster","Segment"]
    ].head()
)

# Cluster Visualization

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

plt.scatter(
    rfm["Frequency"],
    rfm["Monetary"],
    c=rfm["Cluster"]
)

plt.xlabel("Frequency")

plt.ylabel("Monetary")

plt.title(
    "Customer Segmentation"
)

plt.show()

rfm.to_csv(
    "customer_segments.csv"
)

print(
    "Customer segmentation completed"
)