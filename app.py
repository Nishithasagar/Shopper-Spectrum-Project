import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load Datasets

df = pd.read_csv("cleaned_online_retail.csv")

rfm = pd.read_csv("customer_segments.csv")

# Create segment names

rfm["Segment"] = rfm["Cluster"].map({

    0:"High Value",

    1:"Regular",

    2:"Occasional",

    3:"At Risk",

    4:"New Customer"

})

st.set_page_config(
    page_title="Shopper Spectrum",
    layout="wide"
)

# SIDEBAR ONLY
st.sidebar.title(
    "🛒 Shopper Spectrum"
)

st.sidebar.caption(
    "Customer Analytics Dashboard"
)

page = st.sidebar.radio(
    "📌 Navigation",
    [
        "📘 Project Overview",
        "📊 Dashboard",
        "🧹 Data Cleaning",
        "📈 EDA & Insights",
        "🎯 RFM Analysis",
        "📉 Elbow Method",
        "👥 Customer Segmentation",
        "🔗 Product Similarity",
        "🛍 Product Recommendation",
        "👤 Customer Prediction",
        "📌 Final Insights"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Python . VS Code . Streamlit")

# -----------PROJECT OVERVIEW PAGE------------------

if page == "📘 Project Overview":

    st.title(
        "🛒 Shopper Spectrum"
    )

    st.subheader(
        "Customer Analytics & Recommendation System"
    )

    st.divider()

    st.subheader(
        "🎯 Objective"
    )

    st.write(
        """
Analyze customer shopping behaviour,
segment customers using RFM analysis,
and recommend products using similarity analysis.
"""
    )

    st.divider()

    st.subheader(
        "🔄 Project Workflow"
    )

    st.info(
        """
Raw Dataset

↓

Data Cleaning

↓

Exploratory Data Analysis

↓

Customer Segmentation

↓

Product Recommendation

↓

Streamlit Dashboard
"""
    )

    st.divider()

    st.subheader(
        "📦 Modules Included"
    )

    st.success("""
    ✔ Data Cleaning

    ✔ Exploratory Data Analysis

    ✔ Customer Segmentation

    ✔ Product Recommendation

    ✔ Customer Prediction
    """
    )

    st.divider()

    st.subheader(
        "📂 Dataset"
    )

    st.code(
        "online_retail.csv"
    )

#-------------DASHBOARD PAGE----------------

elif page == "📊 Dashboard":

    st.title(
        "📊 Dashboard"
    )

    col1,col2,col3,col4,col5=st.columns(5)

    with col1:
        st.metric(
            "Revenue",
            "₹ 89.1L"
        )

    with col2:
        st.metric(
            "Customers",
            "4338"
        )

    with col3:
        st.metric(
            "Orders",
            "18532"
        )

    with col4:
        st.metric(
            "Products",
            "3877"
        )

    with col5:
        st.metric(
            "Silhouette",
            "0.616"
        )

    st.divider()

    st.subheader(
        "🧹 Data Quality"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Missing Values",
        df.isnull().sum().sum()
    )

    c2.metric(
        "Duplicate Rows",
        df.duplicated().sum()
    )

    c3.metric(
        "Columns",
        df.shape[1]
    )

    st.divider()

    st.subheader(
        "📦 Dataset Structure"
    )

    dtype = (
        df.dtypes
        .astype(str)
        .value_counts()
    )

    st.bar_chart(
        dtype
    )

    st.info(
    f"""
    Rows : {len(df)}

    Columns : {df.shape[1]}

    Countries : {df['Country'].nunique()}
    """
    )

    st.divider()
    
    st.subheader(
        "✅ Data Summary"
    )

    st.success("""
    ✔ Missing values removed

    ✔ Duplicate rows removed

    ✔ Invalid records cleaned
    """
    )

    st.divider()

    st.subheader(
        "📋 Dataset Preview"
    )

    st.dataframe(
        df.head(10)
    )

    st.divider()

    c1,c2 = st.columns(2)

    with c1:

        st.subheader(
            "🌍 Top Countries"
        )

        country = (
            df["Country"]
            .value_counts()
            .head(10)
        )

        st.bar_chart(
            country
        )

    with c2:

        st.subheader(
            "📈 Monthly Sales"
        )

    # Create TotalPrice if missing

        if "TotalPrice" not in df.columns:

            df["TotalPrice"] = (
                df["Quantity"]
                *
                df["UnitPrice"]
            )

    # Convert date

        df["InvoiceDate"] = pd.to_datetime(
            df["InvoiceDate"]
        )

    # Create month

        df["Month"] = (
            df["InvoiceDate"]
            .dt.month_name()
        )

    # Monthly sales

        sales = (

            df.groupby(
                "Month"
            )

            ["TotalPrice"]

            .sum()

        )

        st.line_chart(
            sales
        )

    
    st.divider()

    
#----------------- DATA CLEANING PAGE-----------------

elif page == "🧹 Data Cleaning":

    st.title("🧹 Data Cleaning")

    c1,c2,c3,c4=st.columns(4)

    with c1:
        st.metric("Original Rows","541,909")

    with c2:
        st.metric("Missing Removed","135,080")

    with c3:
        st.metric("Duplicates Removed","5,225")

    with c4:
        st.metric("Final Rows","392,692")

    st.divider()

    st.subheader("Cleaning Steps")

    st.success("""
✔ Missing Values Removed

✔ Duplicate Records Removed

✔ Cancelled Orders Removed

✔ Invalid Quantity Removed

✔ Invalid Price Removed
""")

    st.divider()

    st.subheader("Output File")

    st.code(
        "cleaned_online_retail.csv"
    )

# EDA ANALYSIS

# ---------------- EDA & INSIGHTS ---------------- 

elif page == "📈 EDA & Insights":

    st.title(
        "📈 Exploratory Data Analysis"
    )

# TOP PRODUCTS  

    top_products = (

        df.groupby(
            "Description"
        )["Quantity"]

        .sum()

        .sort_values(
            ascending=False
        )

        .head(10)

    )

    st.subheader(
        "🛒 Top Selling Products"
    )

    st.bar_chart(
        top_products
    )

# COUNTRY 

    country_sales = (

        df.groupby(
            "Country"
        )["TotalAmount"]

        .sum()

        .sort_values(
            ascending=False
        )

        .head(10)

    )

    st.subheader(
        "🌍 Country Wise Revenue"
    )

    st.bar_chart(
        country_sales
    )

# MONTHLY SALES 

    df["Month"] = (

        pd.to_datetime(
            df["InvoiceDate"]
        )

        .dt.month

    )

    monthly = (

        df.groupby(
            "Month"
        )["TotalAmount"]

        .sum()

    )

    st.subheader(
        "📅 Monthly Sales Trend"
    )

    st.line_chart(
        monthly
    )

# CUSTOMER 

    customer = (

        df.groupby(
            "CustomerID"
        )["Quantity"]

        .sum()

        .sort_values(
            ascending=False
        )

        .head(10)

    )

    st.subheader(
        "👤 Most Active Customers"
    )

    st.bar_chart(
        customer
    )

# INSIGHTS 

    st.subheader(
        "🔍 Key Insights"
    )

    st.success(
        """
• Top products contribute most transactions

• Revenue is concentrated in a few countries

• Monthly sales show customer purchase patterns

• Customer activity varies significantly

• K=5 selected for segmentation
"""
    )

#------------------ RFM ANALYSIS-----------------------

elif page == "🎯 RFM Analysis":

    st.title("🎯 RFM Analysis")

    import pandas as pd
    import plotly.express as px

    segment_df = pd.read_csv(
        "customer_segments.csv"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        fig = px.histogram(
            segment_df,
            x="Recency",
            title="Recency"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.histogram(
            segment_df,
            x="Frequency",
            title="Frequency"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col3:

        fig = px.histogram(
            segment_df,
            x="Monetary",
            title="Monetary"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader(
        "📌 RFM Insights"
    )

    st.info("""
✔ Most customers purchase occasionally.

✔ Few customers generate most revenue.

✔ Frequency is concentrated among fewer customers.

✔ Monetary values show high-value customer groups.
""")
    
# ELBOW METHOD

# ---------------- ELBOW METHOD ----------------

elif page == "📉 Elbow Method":

    st.title("📉 Elbow Method")

    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    segment_df = pd.read_csv(
        "customer_segments.csv"
    )

    X = segment_df[
        ["Recency",
         "Frequency",
         "Monetary"]
    ]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    inertia = []

    k_values = range(2,11)

    for k in k_values:

        model = KMeans(
            n_clusters=k,
            random_state=42
        )

        model.fit(
            X_scaled
        )

        inertia.append(
            model.inertia_
        )

    fig, ax = plt.subplots(
        figsize=(10,5)
    )

    ax.plot(
        k_values,
        inertia,
        marker="o"
    )

    ax.set_title(
        "Elbow Method"
    )

    ax.set_xlabel(
        "Clusters"
    )

    ax.set_ylabel(
        "Inertia"
    )

    st.pyplot(
        fig
    )

    st.success(
        "Optimal cluster selected: K = 5"
    )

    st.info("""
Silhouette Score: 0.616

Cluster count chosen after elbow point analysis.
""")

# ---------------- CUSTOMER SEGMENTATION ---------------- 

elif page == "👥 Customer Segmentation":

    import plotly.express as px

    st.title(
        "👥 Customer Segmentation"
    )

# KPIs

    c1,c2,c3,c4=st.columns(4)

    c1.metric(
        "Customers",
        len(rfm)
    )

    c2.metric(
        "Clusters",
        rfm["Cluster"].nunique()
    )

    c3.metric(
        "Avg Frequency",
        round(
            rfm["Frequency"].mean()
        )
    )

    c4.metric(
        "Avg Spend",
        round(
            rfm["Monetary"].mean()
        )
    )

# Cluster Scatter

    st.subheader(
        "📊 Customer Cluster Visualization"
    )

    fig_cluster = px.scatter(

        rfm,

        x="Frequency",

        y="Monetary",

        color="Segment",

        hover_data=[

            "Recency"

        ]

    )

    st.plotly_chart(
        fig_cluster,
        use_container_width=True
    )

# Cluster Count

    st.subheader(
        "📊 Segment Distribution"
    )

    cluster_count = (

        rfm["Cluster"]

        .value_counts()

    )

    st.bar_chart(
        cluster_count
    )

# Revenue

    st.subheader(
        "💰 Revenue by Segment"
    )

    segment_rev = (

        rfm.groupby(
            "Cluster"
        )

        ["Monetary"]

        .sum()

    )

    st.bar_chart(
        segment_rev
    )

# Insights

    st.subheader(
        "🔍 Insights"
    )

    st.info(
        """
• High Value customers spend more

• Regular customers purchase frequently

• Occasional customers buy less

• At-Risk customers need retention

• New Customers show growth opportunity
"""
    )

# ---------------- PRODUCT SIMILARITY ----------------

elif page == "🔗 Product Similarity":

    st.title(
        "🔗 Product Similarity"
    )

# Load similarity matrix

    similarity_df = pd.read_csv(
        "product_similarity.csv",
        index_col=0
    )

# KPIs

    c1,c2,c3 = st.columns(3)

    c1.metric(
        "Products",
        similarity_df.shape[0]
    )

    c2.metric(
        "Similarity Rows",
        similarity_df.shape[0]
    )

    c3.metric(
        "Similarity Columns",
        similarity_df.shape[1]
    )

    st.divider()

# Preview

    st.subheader(
        "📋 Similarity Matrix Preview"
    )

    st.dataframe(
        similarity_df.head(10)
    )

# Heatmap

    import matplotlib.pyplot as plt

    st.subheader(
        "🔥 Product Similarity Heatmap"
    )

    fig, ax = plt.subplots(
        figsize=(10,6)
    )

    sample = similarity_df.iloc[
        :20,
        :20
    ]

    heat = ax.imshow(
        sample
    )

    plt.colorbar(
        heat
    )

    st.pyplot(
        fig
    )

# Product Similarity Search

    st.subheader(
        "🔍 Find Similar Products"
    )

    product = st.text_input(
        "Enter Product Name"
    )

    if st.button(
        "Show Similar"
    ):

        match = [

            x for x in similarity_df.columns

            if product.lower()

            in x.lower()

        ]

        if len(match):

            selected = match[0]

            rec = (

                similarity_df[
                    selected
                ]

                .sort_values(
                    ascending=False
                )

                [1:6]

                .index

                .tolist()

            )

            st.success(
                f"Selected: {selected}"
            )

            for p in rec:

                st.write(
                    "•",
                    p
                )

        else:

            st.error(
                "Product not found"
            )

# Insights

    st.subheader(
        "🔍 Insights"
    )

    st.info(
        """
• Products with higher values are more frequently purchased together

• Similarity matrix powers recommendation engine

• Product relationships are generated using cosine similarity

• Customers buying similar items improve recommendation quality
"""
    )

#-----------------------PRODUCT RECOMMENDATION-----------------------

elif page == "🛍 Product Recommendation":

    st.header(
        "Product Recommendation"
    )

    import pandas as pd

    similarity_df = pd.read_csv(
        "product_similarity.csv",
        index_col=0
    )

    product = st.text_input(
        "Enter Product Name"
    )

    if st.button(
        "Recommend"
    ):

        matches = [

            col

            for col in similarity_df.columns

            if product.lower()
            in col.lower()

        ]

        if matches:

            selected = matches[0]

            rec = (

                similarity_df[
                    selected
                ]

                .sort_values(
                    ascending=False
                )[1:6]

                .index

            )

            st.success(
                f"Selected: {selected}"
            )

            for r in rec:

                st.write(
                    "•",
                    r
                )

        else:

            st.error(
                "Product not found"
            )

# -----------------CUSTOMER PREDICTION---------------------

elif page == "👤 Customer Prediction":

    st.header(
        "Customer Segment Prediction"
    )

    recency = st.number_input(
        "Recency",
        min_value=0
    )

    frequency = st.number_input(
        "Frequency",
        min_value=0
    )

    monetary = st.number_input(
        "Monetary",
        min_value=0.0
    )

    if st.button(
        "Predict"
    ):

        if (
            recency < 30
            and frequency > 100
            and monetary > 5000
        ):

            segment = "High-Value"

        elif (
            recency < 90
            and frequency > 50
        ):

            segment = "Regular"

        elif (
            recency > 180
        ):

            segment = "At-Risk"

        else:

            segment = "Occasional"

        st.success(
            f"Predicted Segment: {segment}"
        )

#------------------ Final Insights -----------------------

elif page == "📌 Final Insights":

    st.header(
        "📌 Final Insights"
    )

    st.success(
        """
• Revenue concentrated among repeat customers

• High-Value segment contributes most sales

• Some countries dominate transactions

• Product recommendation improves cross-selling
"""
    )
