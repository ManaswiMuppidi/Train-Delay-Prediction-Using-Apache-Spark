from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, to_date, count, when,
    isnan, min, max, avg
)

spark = SparkSession.builder \
    .appName("TrainDelayPreprocessing") \
    .master("local[*]") \
    .getOrCreate()

# Load dataset
df = spark.read.csv(
    "train_delay.csv",
    header=True,
    inferSchema=True
)

# Convert date
df = df.withColumn(
    "date",
    to_date(col("date"), "dd-MM-yyyy")
)

print("DATA TYPES AFTER CONVERSION")
df.printSchema()

# Missing values
print("MISSING VALUES")
df.select([
    count(when(col(c).isNull(), c)).alias(c)
    for c in df.columns
]).show()

# Duplicate records
print("DUPLICATE RECORDS")
print("Total:", df.count() - df.dropDuplicates().count())

# Delay statistics
print("DELAY STATISTICS")
df.select(
    min("delay").alias("Minimum"),
    max("delay").alias("Maximum"),
    avg("delay").alias("Average")
).show()

spark.stop()
