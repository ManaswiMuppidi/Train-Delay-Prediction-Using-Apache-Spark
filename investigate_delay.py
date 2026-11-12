from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when

spark = SparkSession.builder \
    .appName("TrainDelayInvestigation") \
    .master("local[*]") \
    .getOrCreate()

df = spark.read.csv(
    "train_delay.csv",
    header=True,
    inferSchema=True
)

# Convert date
from pyspark.sql.functions import to_date

df = df.withColumn(
    "date",
    to_date(col("date"), "dd-MM-yyyy")
)

# Negative delays
print("NEGATIVE DELAY RECORDS")
df.filter(col("delay") < 0).show(10, truncate=False)

print("NEGATIVE DELAY COUNT:",
      df.filter(col("delay") < 0).count())

# Extremely high delays
print("DELAYS ABOVE 500 MINUTES")
df.filter(col("delay") > 500).show(10, truncate=False)

print("COUNT ABOVE 500:",
      df.filter(col("delay") > 500).count())

# Delay categories
print("DELAY CATEGORY COUNTS")

df.select(
    count(when(col("delay") == 0, 1)).alias("On Time"),
    count(when(col("delay") > 0, 1)).alias("Delayed"),
    count(when(col("delay") < 0, 1)).alias("Early")
).show()

spark.stop()