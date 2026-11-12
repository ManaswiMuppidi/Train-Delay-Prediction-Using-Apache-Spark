
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, to_date, month, dayofweek,
    lag, coalesce, lit, abs as spark_abs
)
from pyspark.sql.window import Window

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator


## ==========================================
# 1. START SPARK
# ==========================================

import os

os.environ["PYSPARK_PYTHON"] = r"C:\TrainDelayProject\venv\Scripts\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\TrainDelayProject\venv\Scripts\python.exe"
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("TrainDelayPredictionMLlibV2") \
    .master("local[2]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("===== TRAIN DELAY PREDICTION USING MLlib V2 =====")
# ==========================================
# 2. LOAD DATASET
# ==========================================

df = spark.read.csv(
    "C:/TrainDelayProject/train_delay.csv",
    header=True,
    inferSchema=True
)


# ==========================================
# 3. DATA PREPROCESSING
# ==========================================

df = df.withColumn(
    "date",
    to_date(col("date"), "dd-MM-yyyy")
)

df = df.filter(
    col("delay").isNotNull() &
    col("date").isNotNull() &
    col("train_no").isNotNull() &
    col("station_no").isNotNull() &
    col("station_name").isNotNull()
)


# ==========================================
# 4. PREVIOUS STATION DELAY
# ==========================================

window_spec = Window.partitionBy(
    "train_no", "date"
).orderBy("station_no")

df = df.withColumn(
    "previous_station_delay",
    lag("delay", 1).over(window_spec)
)

df = df.withColumn(
    "previous_station_delay",
    coalesce(col("previous_station_delay"), lit(0))
)


# ==========================================
# 5. FEATURE ENGINEERING
# ==========================================

df = df.withColumn("train_no", col("train_no").cast("string"))
df = df.withColumn("station_name", col("station_name").cast("string"))
df = df.withColumn("station_no", col("station_no").cast("double"))
df = df.withColumn("month", month(col("date")).cast("double"))
df = df.withColumn("day_of_week", dayofweek(col("date")).cast("double"))
df = df.withColumn("delay", col("delay").cast("double"))

df = df.withColumn(
    "previous_station_delay",
    col("previous_station_delay").cast("double")
)

data = df.select(
    "date",
    "train_no",
    "station_name",
    "station_no",
    "month",
    "day_of_week",
    "previous_station_delay",
    "delay"
)

print("Total records:", data.count())


# ==========================================
# 6. SAMPLE 20% OF RECORDS
# ==========================================

sample_data = data.sample(
    withReplacement=False,
    fraction=0.20,
    seed=42
).cache()

# Materialize the sample once
print("Sampled records:", sample_data.count())


# ==========================================
# 7. CHRONOLOGICAL TRAIN/TEST SPLIT
# ==========================================

train_data = sample_data.filter(
    col("date") < "2025-11-01"
).cache()

test_data = sample_data.filter(
    col("date") >= "2025-11-01"
).cache()

# Materialize both datasets

print("Training records:", train_data.count())

print("Testing records:", test_data.count())

print("Verified testing records:", test_data.count())

# Keep the test date for exported results.
# The model does not use date as a feature.

train_data = train_data.drop("date")


# ==========================================
# 8. CATEGORICAL FEATURE ENCODING
# ==========================================

indexer = StringIndexer(
    inputCols=["train_no", "station_name"],
    outputCols=["train_no_index", "station_name_index"],
    handleInvalid="keep"
)

encoder = OneHotEncoder(
    inputCols=["train_no_index", "station_name_index"],
    outputCols=["train_no_encoded", "station_name_encoded"],
    handleInvalid="keep"
)


# ==========================================
# 9. FEATURE ASSEMBLY
# ==========================================

assembler = VectorAssembler(
    inputCols=[
        "station_no",
        "month",
        "day_of_week",
        "previous_station_delay",
        "train_no_encoded",
        "station_name_encoded"
    ],
    outputCol="features"
)


# ==========================================
# 10. RANDOM FOREST MODEL
# ==========================================

rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="delay",
    numTrees=10,
    maxDepth=5,
    maxBins=32,
    subsamplingRate=0.7,
    seed=42
)


# ==========================================
# 11. CREATE PIPELINE
# ==========================================

pipeline = Pipeline(
    stages=[indexer, encoder, assembler, rf]
)


# ==========================================
# 12. MODEL TRAINING
# ==========================================

print("\n===== MODEL TRAINING =====")

model = pipeline.fit(train_data)

print("Model training completed!")


# ==========================================
# 13. GENERATE TEST PREDICTIONS
# ==========================================

print("\n===== GENERATING PREDICTIONS =====")

predictions = model.transform(test_data).cache()

# Materialize predictions once
print("Verified prediction records:", predictions.count())

predictions.select(
    "date",
    "train_no",
    "station_name",
    "delay",
    "previous_station_delay",
    "prediction"
).show(10, truncate=False)


# ==========================================
# 14. MODEL EVALUATION
# ==========================================

metrics = {}

for metric in ["mae", "rmse", "r2"]:

    evaluator = RegressionEvaluator(
        labelCol="delay",
        predictionCol="prediction",
        metricName=metric
    )

    score = evaluator.evaluate(predictions)

    metrics[metric.upper()] = score

    print(metric.upper() + ":", round(score, 4))


# ==========================================
# 15. EXPORT PREDICTIONS
# ==========================================

output_path = "C:/TrainDelayProject/model_evaluation"

prediction_export = predictions.select(
    "date",
    "train_no",
    "station_name",
    "station_no",
    "delay",
    "prediction"
)

prediction_export.write \
    .mode("overwrite") \
    .option("header", True) \
    .csv(output_path + "/model_predictions")

print("Predictions exported successfully!")


# ==========================================
# 16. EXPORT RESIDUALS
# ==========================================

residuals = prediction_export.withColumn(
    "residual",
    col("delay") - col("prediction")
)

residuals = residuals.withColumn(
    "absolute_error",
    spark_abs(col("residual"))
)

residuals.write \
    .mode("overwrite") \
    .option("header", True) \
    .csv(output_path + "/model_residuals")

print("Residuals exported successfully!")


# ==========================================
# 17. EXPORT EVALUATION METRICS
# ==========================================

metrics_df = spark.createDataFrame(
    [(metrics["MAE"], metrics["RMSE"], metrics["R2"])],
    ["MAE", "RMSE", "R2"]
)

metrics_df.write \
    .mode("overwrite") \
    .option("header", True) \
    .csv(output_path + "/model_metrics")

print("Evaluation metrics exported successfully!")


# ==========================================
# 18. SAVE TRAINED MODEL
# ==========================================

model.write().overwrite().save(
    "C:/TrainDelayProject/train_delay_rf_model_v2"
)

print("\nModel saved successfully!")
print("Evaluation files saved in:", output_path)

print("===== MLlib V2 ANALYSIS COMPLETED =====")

spark.stop()
