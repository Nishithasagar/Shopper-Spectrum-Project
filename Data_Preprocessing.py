import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("online_retail.csv", encoding="ISO-8859-1")

# Preview
print(df.head())

# Shape
print("Shape:", df.shape)

# Check Missing Values

df.isnull().sum()

missing = df.isnull().sum()
print(missing)

# Remove missing Customer_ID

df = df.dropna(subset=["CustomerID"])
print(df.shape)

# Check duplicates
print("Duplicate rows:", df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()

# Check new shape
print("Shape after removing duplicates:", df.shape)

# Count cancelled invoices
cancelled = df["InvoiceNo"].astype(str).str.startswith("C").sum()

print("Cancelled invoices:", cancelled)

# Remove cancelled invoices
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

# Check shape
print("Shape after removing cancelled invoices:", df.shape)

# Remove Quantity <= 0
df = df[df["Quantity"] > 0]

# Remove UnitPrice <= 0
df = df[df["UnitPrice"] > 0]

print("Shape after removing invalid Quantity and Price:")
print(df.shape)

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

print(df.dtypes)

df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

print(df.head())

df.to_csv("cleaned_online_retail.csv", index=False)

print("Cleaned dataset saved")
