from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("TrainDelayPrediction") \
    .master("local[*]") \
    .getOrCreate()

print("Apache Spark Version:", spark.version)

data = [
    (7085, 120),
    (4152, 45),
    (7706, 90)
]

columns = ["train_no", "delay"]

df = spark.createDataFrame(data, columns)

df.show()

spark.stop()