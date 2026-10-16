# Paste this at the end of your Colab notebook after final_from_storage is created.

final_pandas = final_from_storage.toPandas()

csv_path = "/content/seller_marketing_performance.csv"
parquet_path = "/content/seller_marketing_performance.parquet"

final_pandas.to_csv(csv_path, index=False)
final_pandas.to_parquet(parquet_path, index=False)

from google.colab import files
files.download(csv_path)
files.download(parquet_path)
