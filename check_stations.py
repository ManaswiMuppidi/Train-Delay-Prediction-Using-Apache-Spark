from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, countDistinct, min, max
)

spark = SparkSession.builder \
    .appName("StationSequenceCheck") \
    .master("local[*]") \
    .getOrCreate()

df = spark.read.csv(
    "cleaned_train_data",
    header=True,
    inferSchema=True
)

print("STATION NUMBER RANGE")
df.select(
    min("station_no").alias("minimum"),
    max("station_no").alias("maximum"),
    countDistinct("station_no").alias("distinct_numbers")
).show()

print("STATION NUMBER AND NAME MAPPING")

df.groupBy("station_no").agg(
    countDistinct("station_name").alias("station_names")
).orderBy("station_no").show(50)

print("STATION SEQUENCE FOR SAMPLE TRAIN 961")

df.filter(
    col("train_no") == 961
).select(
    "date", "train_no", "station_no", "station_name", "delay"
).orderBy("date", "station_no").show(30, truncate=False)

spark.stop()