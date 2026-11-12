from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

spark = SparkSession.builder \
    .appName("TestSavedModel") \
    .master("local[2]") \
    .getOrCreate()

model = PipelineModel.load(
    "C:/TrainDelayProject/train_delay_rf_model"
)

print("Model loaded successfully!")
print("Number of stages:", len(model.stages))

spark.stop()