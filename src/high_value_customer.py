
import pandas as pd
import numpy as np

from datetime import timedelta


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "data/online-retail.xlsx"

REFERENCE_DATE = pd.Timestamp("2011-09-01")

PREDICTION_DAYS = 90

HIGH_VALUE_PERCENTILE = 0.80


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_dataset(path):

    if isinstance(path, pd.DataFrame):
        df = path.copy()
    elif str(path).lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    print("Original dataset:", df.shape)

    # Normalize column names
    df.columns = df.columns.str.strip()

    # Convert data types
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    df["CustomerID"] = pd.to_numeric(
        df["CustomerID"], errors="coerce"
    )

    df["Quantity"] = pd.to_numeric(
        df["Quantity"], errors="coerce"
    )

    df["UnitPrice"] = pd.to_numeric(
        df["UnitPrice"], errors="coerce"
    )

    # Remove transactions without customer identification
    df = df.dropna(subset=["CustomerID"])

    # Remove invalid records
    df = df.dropna(
        subset=["InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice"]
    )

    # Normalize customer ID
    df["CustomerID"] = df["CustomerID"].astype(int).astype(str)

    # Identify cancellations
    cancellation = (
        df["InvoiceNo"].astype(str).str.startswith("C")
    )

    # For this experiment, retain only positive sales
    df = df[
        (~cancellation) &
        (df["Quantity"] > 0) &
        (df["UnitPrice"] > 0)
    ].copy()

    # Calculate transaction revenue
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    print("Clean dataset:", df.shape)

    return df


# ============================================================
# 2. CUSTOMER FEATURE ENGINEERING
# ============================================================

def build_customer_features(df, reference_date):

    # Only historical transactions are allowed
    historical = df[
        df["InvoiceDate"] < reference_date
    ].copy()

    # --------------------------------------------------------
    # Invoice-level aggregation
    # --------------------------------------------------------

    invoice_data = (
        historical
        .groupby(["CustomerID", "InvoiceNo"])
        .agg(
            OrderValue=("Revenue", "sum"),
            Items=("Quantity", "sum"),
            InvoiceDate=("InvoiceDate", "min")
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Customer-level aggregation
    # --------------------------------------------------------

    features = (
        historical
        .groupby("CustomerID")
        .agg(
            LastPurchase=("InvoiceDate", "max"),
            FirstPurchase=("InvoiceDate", "min"),
            Monetary=("Revenue", "sum"),
            TotalQuantity=("Quantity", "sum"),
            UniqueProducts=("StockCode", "nunique"),
        )
    )

    # Number of distinct orders
    frequency = (
        historical
        .groupby("CustomerID")["InvoiceNo"]
        .nunique()
        .rename("Frequency")
    )

    features = features.join(frequency)

    # Average order value
    avg_order_value = (
        invoice_data
        .groupby("CustomerID")["OrderValue"]
        .mean()
        .rename("AvgOrderValue")
    )

    features = features.join(avg_order_value)

    # Average number of items per order
    avg_items = (
        invoice_data
        .groupby("CustomerID")["Items"]
        .mean()
        .rename("AvgItemsPerOrder")
    )

    features = features.join(avg_items)

    # --------------------------------------------------------
    # Recency and customer tenure
    # --------------------------------------------------------

    features["Recency"] = (
        reference_date - features["LastPurchase"]
    ).dt.days

    features["CustomerTenure"] = (
        reference_date - features["FirstPurchase"]
    ).dt.days

    # --------------------------------------------------------
    # Average days between orders
    # --------------------------------------------------------

    def calculate_average_days(dates):

        unique_dates = (
            pd.Series(dates.unique())
            .sort_values()
        )

        if len(unique_dates) < 2:
            return 0.0

        intervals = unique_dates.diff().dt.total_seconds() / 86400

        return intervals.dropna().mean()

    avg_days = (
        historical
        .groupby("CustomerID")["InvoiceDate"]
        .apply(calculate_average_days)
        .rename("AvgDaysBetweenOrders")
    )

    features = features.join(avg_days)

    # --------------------------------------------------------
    # Spending behavior
    # --------------------------------------------------------

    last_30_days = historical[
        historical["InvoiceDate"] >=
        reference_date - timedelta(days=30)
    ]

    previous_60_days = historical[
        (historical["InvoiceDate"] >=
         reference_date - timedelta(days=90)) &
        (historical["InvoiceDate"] <
         reference_date - timedelta(days=30))
    ]

    spend_last_30 = (
        last_30_days
        .groupby("CustomerID")["Revenue"]
        .sum()
        .rename("SpendLast30Days")
    )

    spend_previous_60 = (
        previous_60_days
        .groupby("CustomerID")["Revenue"]
        .sum()
        .rename("SpendPrevious60Days")
    )

    features = features.join(spend_last_30)
    features = features.join(spend_previous_60)

    # Customers without purchases in these periods
    features["SpendLast30Days"] = (
        features["SpendLast30Days"].fillna(0)
    )

    features["SpendPrevious60Days"] = (
        features["SpendPrevious60Days"].fillna(0)
    )

    # --------------------------------------------------------
    # Country
    # --------------------------------------------------------

    country = (
        historical
        .groupby("CustomerID")["Country"]
        .agg(lambda x: x.mode().iloc[0])
        .rename("Country")
    )

    features = features.join(country)

    # Remove timestamp columns
    features = features.drop(
        columns=["LastPurchase", "FirstPurchase"]
    )

    features = features.reset_index()

    return features


# ============================================================
# 3. BUILD TARGET VARIABLE
# ============================================================

def build_target(df, reference_date, prediction_days):

    prediction_end = (
        reference_date + timedelta(days=prediction_days)
    )

    # Customers must have historical purchase activity
    historical_customers = df[
        df["InvoiceDate"] < reference_date
    ]["CustomerID"].unique()

    # Future transactions
    future = df[
        (df["InvoiceDate"] >= reference_date) &
        (df["InvoiceDate"] < prediction_end) &
        (df["CustomerID"].isin(historical_customers))
    ].copy()

    # Future revenue by customer
    future_revenue = (
        future
        .groupby("CustomerID")["Revenue"]
        .sum()
        .rename("FutureRevenue")
    )

    # Include customers with no future purchases
    target = pd.DataFrame({
        "CustomerID": historical_customers
    })

    target = target.merge(
        future_revenue,
        on="CustomerID",
        how="left"
    )

    target["FutureRevenue"] = (
        target["FutureRevenue"].fillna(0)
    )

    # Calculate the 80th percentile
    threshold = target["FutureRevenue"].quantile(
        HIGH_VALUE_PERCENTILE
    )

    # Binary classification
    target["HighValueCustomer"] = (
        (target["FutureRevenue"] >= threshold) &
        (target["FutureRevenue"] > 0)
    ).astype(int)

    print("\nTarget threshold:", round(threshold, 2))

    print("\nTarget distribution:")
    print(
        target["HighValueCustomer"]
        .value_counts(normalize=True)
    )

    return target, threshold


# ============================================================
# 4. CREATE FINAL DATASET
# ============================================================

def create_ml_dataset(df):

    features = build_customer_features(
        df,
        REFERENCE_DATE
    )

    target, threshold = build_target(
        df,
        REFERENCE_DATE,
        PREDICTION_DAYS
    )

    dataset = features.merge(
        target,
        on="CustomerID",
        how="inner"
    )

    # Remove identifier and future revenue from model inputs
    X = dataset.drop(
        columns=[
            "CustomerID",
            "FutureRevenue",
            "HighValueCustomer"
        ]
    )

    y = dataset["HighValueCustomer"]

    print("\nFinal feature matrix:", X.shape)

    print("\nFeature columns:")
    print(X.columns.tolist())

    return X, y, dataset, threshold


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":

    df = load_dataset(DATASET_PATH)

    X, y, dataset, threshold = create_ml_dataset(df)

    dataset.to_csv(
        "data/high_value_customer_dataset.csv",
        index=False
    )

    print("\nFeature engineering completed!")