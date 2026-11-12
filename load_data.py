from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("TrainDelayDataLoading") \
    .master("local[*]") \
    .getOrCreate()

df = spark.read.csv(
    "C:/TrainDelayProject/train_delay.csv",
    header=True,
    inferSchema=True
)

print("DATASET SCHEMA")
df.printSchema()

print("FIRST 10 RECORDS")
df.show(10, truncate=False)

print("TOTAL RECORDS:", df.count())

print("TOTAL COLUMNS:", len(df.columns))

spark.stop()