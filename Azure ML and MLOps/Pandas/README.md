Daily practice as part of my Azure MLOps and AI-300 preparation. Each day contains a standalone Python script, with no dependencies between exercises.

- Think of Pandas as a digital spreadsheet for Python. It lets you easily load, clean, and organize rows and columns of data.
- So, you can work with them like you would in Excel.

When you need to use it?

- Pandas is particularly effective for working with structured or tabular data, such as customer records, transaction data, sensor measurements,   and business metrics.
- It is mainly used to load, explore, clean, validate, transform, and prepare data before machine-learning training.
- For very large datasets that cannot fit into memory, distributed tools such as PySpark, Azure Databricks, or the pandas API on
  Spark are usually more appropriate.
- In an MLOps workflow, pandas is often used to inspect datasets, handle missing values and duplicates, engineer features,
  validate data  quality, split features from labels, and create reusable preprocessing components before training a model
  with scikit-learn or another framework.
- The processed data can then be used in Azure Machine Learning pipelines and registered as a versioned data asset.
