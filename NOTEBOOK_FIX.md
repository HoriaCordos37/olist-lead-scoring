# If the notebook model cell gives an Imputer error

If Spark shows an error like:

```text
surrogate cannot be computed. All the values in declared_product_catalog_size are Null, Nan or missingValue
```

use only these numeric columns for the ML model:

```python
numeric_columns = [
    "declared_monthly_revenue",
    "days_to_close",
]
```

Do not include this column in the model features:

```python
"declared_product_catalog_size"
```

That column may be empty for all rows after the final join, so Spark's `Imputer(strategy="median")` cannot calculate a replacement value.

Then restart the Colab runtime and run the notebook from the top.
