# 3-minute demo script

## 1. Opening

This is my Olist Lead Scoring & Seller Analytics app.

The business problem is:

> Can we use marketing funnel data and e-commerce seller performance data to predict which leads will become high-value sellers?

## 2. Pipeline explanation

The original notebook uses two real Olist datasets:

1. Olist Brazilian e-commerce dataset
2. Olist marketing funnel dataset

The pipeline is:

```text
Kaggle datasets
→ raw CSV storage
→ Spark cleaning
→ processed Parquet files
→ final seller-level dataset
→ machine learning model
→ Streamlit web demo
```

## 3. Actual output

The final output of the notebook is:

```text
seller_marketing_performance.parquet
```

This dataset contains seller-level marketing and performance information.

The model predicts:

```text
high_value_seller
```

Where:

```text
1 = high-value seller
0 = normal-value seller
```

The notebook uses this business rule:

```text
high-value seller = seller revenue >= 3289.80
```

## 4. Show the app pages

### Project Overview

Explain the business goal and the pipeline.

### Data Explorer

Show the final dataset and the columns.

### Business Dashboard

Show:
- seller class distribution
- high-value seller rate by lead origin
- revenue by business segment
- seller state performance

### Prediction Demo

Enter a lead profile and show the prediction.

Example explanation:

> The model predicts whether this lead is likely to become a high-value seller. If the probability is high, the business should prioritize this lead for sales follow-up and onboarding support.

### Model Performance

Show:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- confusion matrix
- feature importance

## 5. Final conclusion

This project shows an end-to-end big data pipeline. It combines advertising data with e-commerce data, processes the data using Spark, saves the final dataset as Parquet, trains a machine learning model, and presents the result in a Streamlit demo app.
