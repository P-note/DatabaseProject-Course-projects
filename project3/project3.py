from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("ClassificationPractice").getOrCreate()

train_df = spark.read.csv("./dataset/train_wine.csv", header=True, inferSchema=True)
test_df = spark.read.csv("./dataset/test_wine.csv", header=True, inferSchema=True)

### 1. Converting & Visualizing the provided dataset in DataFrame 

columns = train_df.columns[1:]

assembler = VectorAssembler(inputCols=columns, outputCol="features")

train_df = assembler.transform(train_df).withColumnRenamed("Type", "label")
test_df = assembler.transform(test_df).withColumnRenamed("Type", "label")

train_df.show()
test_df.show()

### 2-1. K-means Clustering 

kmeans = KMeans(featuresCol="features",
                predictionCol='prediction',
				k=2,
				maxIter=20,
				distanceMeasure='euclidean')
model = kmeans.fit(train_df)
predictions = model.transform(test_df).select("label", "prediction")
output_groupby_columns = predictions.groupBy("label", "prediction")

### 2-2. Multi-class classification 

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

### 3. Logistic Regression 

from pyspark.ml.classification import LogisticRegression
from pyspark.sql.types import DoubleType

train_df = train_df.filter(train_df["label"] != "3.0")
test_df = test_df.filter(test_df["label"] != "3.0")

logistic_regression = LogisticRegression(featuresCol="features", labelCol="label", predictionCol='prediction', maxIter=100)
lr_model = logistic_regression.fit(train_df)
lr_preds = model.transform(test_df)
lr_preds = lr_preds.withColumn("prediction", lr_preds["prediction"].cast(DoubleType()))
lr_accuracy = evaluator.evaluate(lr_preds)


print(f"Accuracy: {lr_accuracy}")


### 4. Decision Tree

from pyspark.ml.classification import DecisionTreeClassifier

seed = 2024

dt=DecisionTreeClassifier(maxDepth=2, featuresCol="features", labelCol="label", predictionCol='prediction')

dt_model = dt.fit(train_df)
dt_preds = dt_model.transform(test_df)
dt_accuracy = evaluator.evaluate(dt_preds)


print(f"Accuracy: {dt_accuracy}")

### 5. SVM

from pyspark.ml.classification import LinearSVC

seed = 2024

from pyspark.sql.functions import when

svm = LinearSVC(featuresCol="features", labelCol="label", predictionCol='prediction', maxIter=100)
# convert train_df, test_df labels to fit in range [0,1]
train_df = train_df.withColumn("label", when(train_df["label"] == 2, 1).otherwise(0))
test_df = test_df.withColumn("label", when(test_df["label"] == 2, 1).otherwise(0))
svm_model = svm.fit(train_df)
svm_preds = svm_model.transform(test_df)
svm_accuracy = evaluator.evaluate(svm_preds)


# print(f"Accuracy: {rfc_accuracy}")
print(f"Accuracy: {svm_accuracy}")

