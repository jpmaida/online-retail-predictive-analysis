# HighValue AI

HighValue AI is a Streamlit application for exploring and predicting which
retail customers are likely to generate high revenue. It uses historical
Online Retail transactions to build customer-level features, trains two
classification models, and provides an interactive dashboard for predictions
and model evaluation.

## How it works

The workflow has three stages:

1. **Prepare the customer dataset.** `src/high_value_customer.py` reads
   `data/online-retail.xlsx`, cleans transaction records, builds customer
   features from transactions before September 1, 2011, and labels customers
   using revenue during the following 90-day period. A customer is labeled
   high value when their future revenue is at or above the 80th percentile
   and is greater than zero. The resulting dataset is written to
   `data/high_value_customer_dataset.csv`.
2. **Train and evaluate the models.** `src/train_model.py` uses that dataset
   to train both classifiers, evaluates them on a stratified 20% holdout set,
   and saves the models and evaluation files.
3. **Explore predictions and evaluation.** `app.py` loads the saved models and
   results and starts the Streamlit dashboard.

The customer features include recency, order frequency, total and average
spend, quantities and products purchased, average time between orders,
customer tenure, recent spending, and country. Customer IDs and future revenue
are excluded from the model inputs.

## Models

Both models use the same preprocessing pipeline:

- Missing numeric values are filled with the median, then features are
  standardized.
- Missing categorical values are filled with the most frequent value, then
  one-hot encoded. Unseen categories are ignored.

The classifiers are:

- **Logistic Regression** — a linear classification baseline with balanced
  class weights and up to 2,000 iterations.
- **Random Forest** — an ensemble of 500 decision trees, with balanced
  subsampling of classes.

The training script ranks the models by **PR-AUC** and saves the fitted
pipelines as `models/LogisticRegression.joblib` and
`models/RandomForest.joblib`.

## Metrics and dashboard

The held-out test data is used to calculate the following metrics for each
model:

| Metric | What it measures |
| --- | --- |
| **Accuracy** | Fraction of all predictions that are correct. |
| **Precision** | Fraction of customers predicted high value who are actually high value. |
| **Recall** | Fraction of actual high-value customers correctly identified. |
| **F1** | Harmonic mean of precision and recall. |
| **ROC-AUC** | How well the model separates the two classes across classification thresholds. |
| **PR-AUC** | Area under the precision-recall curve; useful when the positive class is less common. |

The dashboard's **Model Performance** page displays these metrics, ROC and
precision-recall curves, a confusion matrix (true negatives, false positives,
false negatives, and true positives), and Random Forest feature importance.

The **Customer Prediction** page offers two ways to get a prediction:

- **Manual Customer Input:** enter the customer's historical feature values.
- **Upload Transactions:** upload a CSV or Excel file with transactions for
  exactly one customer. The file must contain the transaction columns used by
  the data preparation script, including `CustomerID`, `InvoiceNo`,
  `InvoiceDate`, `Quantity`, `UnitPrice`, `StockCode`, and `Country`. All
  transactions must be earlier than September 1, 2011.

For either option, both models report the predicted class and the probability
of being high value. The sidebar's **Classification Threshold** controls the
probability cutoff (default `0.50`): a probability at or above the cutoff is
classified as high value. The page also indicates whether the models agree.

## Requirements

Run the commands from the repository root with Python and the project
dependencies installed. The application uses pandas, NumPy, scikit-learn,
joblib, openpyxl, Streamlit, and Plotly. `src/evaluate_model.py` is an optional
standalone plotting script and additionally uses Matplotlib and Seaborn.

The supplied source workbook `data/online-retail.xlsx` must be present for the
first step. The training and application steps create or read files under the
`data/`, `models/`, and `results/` directories.

## Run the application

Execute the commands in order from the project root, waiting for each script
to finish before running the next:

```bash
python src/high_value_customer.py
python src/train_model.py
streamlit run app.py
```

After Streamlit starts, open the local URL printed in the terminal (usually
`http://localhost:8501`).

## Generated files

- `data/high_value_customer_dataset.csv` — engineered customer features and
  target labels.
- `models/LogisticRegression.joblib` and `models/RandomForest.joblib` — trained
  model pipelines.
- `results/model_comparison.csv` — evaluation metrics for both models.
- `results/predictions_LogisticRegression.csv` and
  `results/predictions_RandomForest.csv` — actual labels, predicted labels,
  and probabilities for the held-out test set.
- `results/feature_importance.csv` — Random Forest feature importance scores.

## Optional: predict from the command line

After training, `src/predict_customer.py` can predict from a CSV containing
transactions for one customer:

```bash
python src/predict_customer.py --input data/new_customer.csv
```

The optional `--threshold` argument sets the classification cutoff; it must
be greater than 0 and less than 1. For example:

```bash
python src/predict_customer.py --input data/new_customer.csv --threshold 0.65
```
