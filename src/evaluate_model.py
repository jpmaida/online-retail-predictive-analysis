
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)


df = pd.read_csv("data/high_value_customer_dataset.csv")

X = df.drop(
    columns=[
        "CustomerID",
        "FutureRevenue",
        "HighValueCustomer"
    ]
)

y = df["HighValueCustomer"]

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = joblib.load("models/RandomForest.joblib")

probabilities = model.predict_proba(X_test)[:, 1]

predictions = model.predict(X_test)


# ------------------------------------------------------------
# 1. Confusion Matrix
# ------------------------------------------------------------

cm = confusion_matrix(y_test, predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Regular", "High Value"]
)

disp.plot(cmap="Blues")

plt.title("High Value Customer - Confusion Matrix")

plt.show()


# ------------------------------------------------------------
# 2. ROC Curve
# ------------------------------------------------------------

RocCurveDisplay.from_predictions(
    y_test,
    probabilities
)

plt.title("High Value Customer - ROC Curve")

plt.show()


# ------------------------------------------------------------
# 3. Precision-Recall Curve
# ------------------------------------------------------------

PrecisionRecallDisplay.from_predictions(
    y_test,
    probabilities
)

plt.title("High Value Customer - Precision Recall")

plt.show()


# ------------------------------------------------------------
# 4. Feature Importance
# ------------------------------------------------------------

importance = pd.read_csv("results/feature_importance.csv")

importance = importance.head(15).sort_values(
    "Importance"
)

plt.figure(figsize=(10, 7))

sns.barplot(
    data=importance,
    x="Importance",
    y="Feature"
)

plt.title("Top 15 Features - Random Forest")

plt.tight_layout()

plt.show()
