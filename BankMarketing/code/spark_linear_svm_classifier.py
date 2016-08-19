from pyspark import SparkConf, SparkContext
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.regression import LabeledPoint
from numpy import array
import math
import time

#building Spark Configuration and Getiing a Spark Context and Loading Data into an RDD
conf = (SparkConf().setMaster("local[8]").setAppName("bank_marketing_classification_linear_svm").set("spark.executor.memory", "1g"))
sc = SparkContext(conf = conf)

testing_data = sc.textFile("../data/bank-additional-test-normalized.csv")
training_data = sc.textFile("../data/bank-additional-train-normalized.csv")

#parsing mapper function
def mapper_CF(x):
	z = str(x)
	tokens = z.split(',')
	c = len(tokens)
	if c != 21:
		print "*********DATA VALIDATION ERROR\n***********8"
	attrib = [float(y) for y in tokens]
	return LabeledPoint(attrib[20],attrib[0:20])

vectorize_start = time.time()
vectorized_data = training_data.map(mapper_CF)
vectorized_testing_data = testing_data.map(mapper_CF)
vectorize_end = time.time()
print "******************VECTORIZING: DONE********************"

#building a logistic regression training model
train_start = time.time()
model = SVMWithSGD.train(vectorized_data)
train_end = time.time()
print "******************MODEL TRAINING: DONE********************"

#predicting classes for testing data and evaluating
def mapper_predict(x):
	predicted_class = model.predict(x.features)
	#predicted_class = int(round(predicted_class))
	actual_class = x.label
	return (actual_class, predicted_class)

pred_start = time.time()
actual_and_predicted = vectorized_testing_data.cache().map(mapper_predict)
pred_end = time.time()
print "******************PREDICTION: DONE********************"

#evaluation
eval_start = time.time()
training_error = actual_and_predicted.filter(lambda (a, p): a != p).count() / float(actual_and_predicted.count())
MSE = actual_and_predicted.map(lambda (a, p): (a - p)**2).reduce(lambda x, y: x + y) / actual_and_predicted.count()
RMSE = math.sqrt(MSE)
eval_end = time.time()
print "******************EVALUATION: DONE********************"

#efficiency: time calculation
vectorize_time = vectorize_end - vectorize_start
train_time = train_end - train_start
pred_time = pred_end - pred_start
eval_time = eval_end - eval_start
print "******************TIME CALCULATION: DONE********************"

print "******************RESULTS********************"
title = "***SPARK LINEAR SVM CLASSIFIER RESULTS***\n"
accuracy = "##############Accuracy##################\n"
train_err_res = ("Training Error = " + str(training_error)) + '\n'
rmse_res = ("Mean Squared Error = " + str(RMSE)) + '\n'

efficiency = "##############Efficiency################\n"
vectorize_res = ("Vectorizing Time = " + str(vectorize_time)) + '\n'
train_res = ("Training TIme = " + str(train_time)) + '\n'
pred_res = ("Predicting TIme = " + str(pred_time)) + '\n'
eval_res = ("Evaluation TIme = " + str(eval_time)) + '\n'

result = title + accuracy + train_err_res + rmse_res + efficiency + vectorize_res + train_res + pred_res + eval_res

res_fh = open('../spark_result/linear_svm_classifier_result.txt','w')
res_fh.write(result)
res_fh.close()

print title
print "\n##############Accuracy##################"
print("Training Error (Mean Absolute Error) = " + str(training_error))
print("Root Mean Squared Error = " + str(RMSE))
print "########################################"
print "\n##############Efficiency################"
print("Vectorizing Time = " + str(vectorize_time))
print("Training TIme = " + str(train_time))
print("Predicting TIme = " + str(pred_time))
print("Evaluation TIme = " + str(eval_time))
print "########################################"
