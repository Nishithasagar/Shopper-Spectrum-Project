import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(
    "cleaned_online_retail.csv",
    parse_dates=["InvoiceDate"]
)

print(df.shape)

# Transaction Volume by Country (Top 10)
country = (
    df["Country"]
    .value_counts()
    .head(10)
)

plt.figure(figsize=(10,5))

country.plot(kind="bar")

plt.title("Top 10 Countries by Transactions")
plt.xlabel("Country")
plt.ylabel("Transactions")

plt.xticks(rotation=45)

plt.show()

# Top 10 Selling Products
top_products = (
    df.groupby("Description")["Quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12,5))

top_products.plot(kind="bar")

plt.title("Top Selling Products")

plt.xticks(rotation=90)

plt.show()

# Purchase Trend Over Time
daily_sales = (
    df.groupby(
        df["InvoiceDate"].dt.date
    )["TotalAmount"]
    .sum()
)

plt.figure(figsize=(12,5))

daily_sales.plot()

plt.title("Purchase Trend Over Time")

plt.ylabel("Revenue")

plt.show()

# Monetary Distribution

plt.figure(figsize=(8,5))

sns.histplot(
    df["TotalAmount"],
    bins=50
)

plt.title("Transaction Amount Distribution")

plt.show()

# Create RFM table

import pandas as pd

df = pd.read_csv(
    "cleaned_online_retail.csv",
    parse_dates=["InvoiceDate"]
)

reference_date = df["InvoiceDate"].max()

rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (reference_date - x.max()).days,
    "InvoiceNo": "count",
    "TotalAmount": "sum"
})

rfm.columns = [
    "Recency",
    "Frequency",
    "Monetary"
]

print(rfm.head())
print(rfm.shape)

# RFM Distribution

import matplotlib.pyplot as plt

rfm.hist(
    figsize=(12,6),
    bins=30
)

plt.show()

# Standardize data

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

rfm_scaled = scaler.fit_transform(rfm)

print(rfm_scaled.shape)

# Elbow curve(Cluster Section)

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertia = []

for k in range(2,11):

    model = KMeans(
        n_clusters=k,
        random_state=42
    )

    model.fit(rfm_scaled)

    inertia.append(model.inertia_)

plt.figure(figsize=(8,5))

plt.plot(
    range(2,11),
    inertia,
    marker="o"
)

plt.title("Elbow Method")

plt.xlabel("Number of Clusters")

plt.ylabel("Inertia")

plt.show()

# Silhouette Score

from sklearn.metrics import silhouette_score

for k in range(2,11):

    km = KMeans(
        n_clusters=5,
        random_state=42
    )

    labels = km.fit_predict(
        rfm_scaled
    )

    score = silhouette_score(
        rfm_scaled,
        labels
    )

    print(
        f"K={k}, Score={score:.3f}"
    )

    