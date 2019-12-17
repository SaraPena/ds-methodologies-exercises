import pyspark

spark = pyspark.sql.SparkSession.builder.getOrCreate()
spark.range(10).show()


