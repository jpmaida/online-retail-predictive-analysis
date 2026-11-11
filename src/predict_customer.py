
import argparse
import joblib
import pandas as pd

from pathlib import Path

from high_value_customer import (
    load_dataset,
    build_customer_features,
    REFERENCE_DATE
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATHS = {
    "Logistic Regression": "models/LogisticRegression.joblib",
    "Random Forest": "models/RandomForest.joblib"
}

DEFAULT_THRESHOLD = 0.50


# ============================================================
# 1. LOAD CUSTOMER TRANSACTIONS
# ============================================================

def load_customer_data(csv_path):

    df = load_dataset(csv_path)

    customers = df["CustomerID"].unique()

    if len(customers) != 1:
        raise ValueError(
            "The input CSV must contain exactly one customer."
        )

    customer_id = customers[0]

    # Ensure that all transactions belong to the
    # historical observation window
    if not (df["InvoiceDate"] < REFERENCE_DATE).all():
        raise ValueError(
            f"Transactions must be earlier than {REFERENCE_DATE}."
        )

    return df, customer_id


# ============================================================
# 2. GENERATE FEATURES
# ============================================================

def generate_customer_features(df):

    features = build_customer_features(
        df,
        REFERENCE_DATE
    )

    if len(features) != 1:
        raise ValueError(
            "Could not generate features for exactly one customer."
        )

    customer_id = features.iloc[0]["CustomerID"]

    X = features.drop(
        columns=["CustomerID"]
    )

    return customer_id, X


# ============================================================
# 3. LOAD TRAINED MODELS
# ============================================================

def load_models():

    models = {}

    for model_name, model_path in MODEL_PATHS.items():

        path = Path(model_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Model not found: {path}"
            )

        models[model_name] = joblib.load(path)

    return models


# ============================================================
# 4. PREDICT CUSTOMER VALUE
# ============================================================

def predict_customer(X, models, threshold):

    results = []

    for model_name, model in models.items():

        # Align features with the original training schema
        expected_features = model.feature_names_in_

        missing_features = set(expected_features) - set(X.columns)

        if missing_features:
            raise ValueError(
                f"Missing features for {model_name}: "
                f"{missing_features}"
            )

        customer_X = X.loc[:, expected_features]

        # Probability of belonging to class 1
        probabilities = model.predict_proba(customer_X)[0]

        class_labels = list(model.classes_)

        positive_index = class_labels.index(1)

        probability = probabilities[positive_index]

        # Apply configurable decision threshold
        prediction = int(probability >= threshold)

        classification = (
            "HIGH VALUE CUSTOMER"
            if prediction == 1
            else "REGULAR CUSTOMER"
        )

        results.append({
            "Model": model_name,
            "Prediction": prediction,
            "Classification": classification,
            "Probability": probability
        })

    return results


# ============================================================
# 5. DISPLAY RESULTS
# ============================================================

def display_results(customer_id, X, results, threshold):

    print("\n" + "=" * 65)
    print("HIGH VALUE CUSTOMER - INFERENCE")
    print("=" * 65)

    print(f"\nCustomer ID: {customer_id}")

    print("\nCustomer features:")

    for feature, value in X.iloc[0].items():

        if isinstance(value, float):
            print(f"  {feature}: {value:.2f}")
        else:
            print(f"  {feature}: {value}")

    print("\n" + "-" * 65)

    print(f"Decision threshold: {threshold:.0%}")

    for result in results:

        print("\nModel:", result["Model"])

        print(
            "Classification:",
            result["Classification"]
        )

        print(
            "Probability of High Value:",
            f"{result['Probability']:.2%}"
        )

        print(
            "Predicted class:",
            result["Prediction"]
        )

    print("\n" + "=" * 65)

    # Model agreement
    predictions = [
        result["Prediction"]
        for result in results
    ]

    if len(set(predictions)) == 1:

        print("CONSENSUS: Both models agree.")

    else:

        print("DISAGREEMENT: Models produced different predictions.")

    print("=" * 65)


# ============================================================
# 6. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="High Value Customer Prediction"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="CSV file containing customer transactions"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Classification probability threshold (default: 0.50)"
    )

    args = parser.parse_args()

    if not 0 < args.threshold < 1:
        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    # Load transactions
    df, _ = load_customer_data(args.input)

    # Generate features
    customer_id, X = generate_customer_features(df)

    # Load models
    models = load_models()

    # Run inference
    results = predict_customer(
        X,
        models,
        args.threshold
    )

    # Display results
    display_results(
        customer_id,
        X,
        results,
        args.threshold
    )


if __name__ == "__main__":
    main()
