from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, when

spark = SparkSession.builder \
    .appName("TrainDelayDataCleaning") \
    .master("local[*]") \
    .getOrCreate()

# Load original dataset
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

# Flag unusually high delays
df = df.withColumn(
    "extreme_delay",
    when(col("delay") > 500, 1).otherwise(0)
)

# Separate records with valid delay values
valid_df = df.filter(col("delay").isNotNull())

# Save cleaned dataset
valid_df.write \
    .mode("overwrite") \
    .option("header", True) \
    .csv("cleaned_train_data")

print("Original records:", df.count())
print("Valid delay records:", valid_df.count())
print("Missing delay records:", df.filter(col("delay").isNull()).count())
print("Extreme delay records:", valid_df.filter(col("extreme_delay") == 1).count())

print("Cleaned dataset sample:")
valid_df.show(10, truncate=False)

spark.stop()