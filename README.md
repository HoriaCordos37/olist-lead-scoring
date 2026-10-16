# Olist Lead Scoring & Seller Analytics

This is a Streamlit demo app for the **Olist Lead Scoring & Seller Analytics**.

The app demonstrates:

- the final integrated seller dataset
- business dashboard charts
- high-value seller prediction
- model performance metrics
- a live prediction form for new marketing leads

## Recommended app outside Google Colab

Use:

```text
VS Code + Streamlit
```

Google Colab is good for running the PySpark notebook. Streamlit is better for presenting the final demo as a web app.

## How to run locally

Open this folder in VS Code, then run:

```bash
python -m venv .venv
```

Activate the environment:

### Windows

```bash
.venv\Scripts\activate
```

### Mac/Linux

```bash
source .venv/bin/activate
```

Install the requirements:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

## How to use your real Colab output

The app already includes sample demo data so it runs immediately.

For the real final project demo:

1. Run your notebook in Google Colab until the final dataset is created.
2. Export the final dataset from Colab.
3. Put the exported file here:

```text
data/seller_marketing_performance.parquet
```

or:

```text
data/seller_marketing_performance.csv
```

4. Run:

```bash
streamlit run app.py
```

The app will automatically use your real dataset instead of the included demo sample.

## Colab export cell

After running your notebook, add this cell at the end:

```python
# Export final dataset for Streamlit
final_pandas = final_from_storage.toPandas()

csv_path = "/content/seller_marketing_performance.csv"
parquet_path = "/content/seller_marketing_performance.parquet"

final_pandas.to_csv(csv_path, index=False)
final_pandas.to_parquet(parquet_path, index=False)

from google.colab import files
files.download(csv_path)
files.download(parquet_path)
```

Download one of those files and place it in the `data` folder.

## Final demo talking points

Use this explanation:

> This project combines Olist e-commerce data with Olist marketing funnel data. The data is processed with Spark, saved as Parquet, and integrated at seller level using seller_id. Then a machine learning model predicts whether a marketing lead becomes a high-value seller. The Streamlit app presents the pipeline, explores the final dataset, shows business insights, evaluates the model, and allows live prediction for new leads.

## Important note

The included `demo_seller_marketing_performance.csv` is sample data created only so the app runs immediately. Replace it with your real Colab export before submitting or presenting the final version.
