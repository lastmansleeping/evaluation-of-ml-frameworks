from sklearn import svm
import numpy as np
import math
import time


testing_data = open("../data/higgs-testing.csv")
training_data = open("../data/higgs.csv")

#parsing mapper function
def mapper_CF(x):
	z = str(x)
	tokens = z.split(',')
	c = len(tokens)
	if c != 29:
		print "*********DATA VALIDATION ERROR\n***********8"
	attrib = [float(y) for y in tokens]
	x = attrib[1:29]
	y = attrib[0]
	return (x,y)

def vectorize(fh):
	features = []
	targets = []
	for line in fh:
		(xi,yi) = mapper_CF(line)
		xi = np.array(xi)
		features.append(xi)
		targets.append(yi)
	
	X = np.array(features)
	y = np.array(targets)
	return (X,y)

vectorize_start = time.time()
vectorized_data = vectorize(training_data)
vectorized_testing_data = vectorize(testing_data)
training_data.close()
testing_data.close()
vectorize_end = time.time()
print "******************VECTORIZING: DONE********************"

#building a logistic regression training model
train_start = time.time()
model = svm.LinearSVC()
model = model.fit(vectorized_data[0], vectorized_data[1])
train_end = time.time()
print "******************MODEL TRAINING: DONE********************"

#predicting classes for testing data and evaluating
def mapper_predict(x):
	predicted_class = model.predict(x.features)
	#predicted_class = int(round(predicted_class))
	actual_class = x.label
	return (actual_class, predicted_class)

pred_start = time.time()
predicted = model.predict(vectorized_testing_data[0])
actual = vectorized_testing_data[1]
actual_and_predicted = np.transpose(np.vstack((actual,predicted)))
actual_and_predicted = actual_and_predicted.tolist()
pred_end = time.time()
print "******************PREDICTION: DONE********************"

#evaluation
eval_start = time.time()
samples_count = len(actual_and_predicted)
training_error = len(filter(lambda (a, p): a != p, actual_and_predicted)) / float(samples_count)
MSE_map = map(lambda (a, p): (a - p)**2,actual_and_predicted)
MSE_reduce = reduce(lambda x, y: x + y, MSE_map)
MSE = MSE_reduce / float(samples_count)
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
title = "***SEQUENTIAL LINEAR SVM CLASSIFIER RESULTS***\n"
accuracy = "##############Accuracy##################\n"
train_err_res = ("Training Error = " + str(training_error)) + '\n'
rmse_res = ("Mean Squared Error = " + str(RMSE)) + '\n'

efficiency = "##############Efficiency################\n"
vectorize_res = ("Vectorizing Time = " + str(vectorize_time)) + '\n'
train_res = ("Training TIme = " + str(train_time)) + '\n'
pred_res = ("Predicting TIme = " + str(pred_time)) + '\n'
eval_res = ("Evaluation TIme = " + str(eval_time)) + '\n'

result = title + accuracy + train_err_res + rmse_res + efficiency + vectorize_res + train_res + pred_res + eval_res

res_fh = open('../result/sequential_linear_svm_classifier_result.txt','w')
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
