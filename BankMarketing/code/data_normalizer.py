from pyspark import SparkConf, SparkContext
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
from numpy import array
import math
import time
#attributes: 21 in total where categorical: 10, numeric: 10, binary: 1

job = {}
job['admin.'] = 0
job['blue-collar'] = 1
job['entrepreneur'] = 2
job['housemaid'] = 3
job['management'] = 4
job['retired'] = 5
job['self-employed'] = 6
job['services'] = 7
job['student'] = 8
job['technician'] = 9
job['unemployed'] = 10
job['unknown'] = -1

marital = {}
marital['divorced'] = 1
marital['married'] = 2
marital['single'] = 3
marital['unknown'] = -1

education = {}
education['basic.4y'] = 1
education['basic.6y'] = 2
education['basic.9y'] = 3
education['high.school'] = 4
education['illiterate'] = 5
education['professional.course'] = 6
education['university.degree'] = 7
education['unknown'] = -1

default = {}
default['no'] = 0
default['yes'] = 1
default['unknown'] = -1

housing = {}
housing['no'] = 0
housing['yes'] = 1
housing['unknown'] = -1

loan = {}
loan['no'] = 0
loan['yes'] = 1
loan['unknown'] = -1

contact = {}
contact['cellular'] = 1
contact['telephone'] = 2

month = {}
month['jan'] = 1
month['feb'] = 2
month['mar'] = 3
month['apr'] = 4
month['may'] = 5
month['jun'] = 6
month['jul'] = 7
month['aug'] = 8
month['sep'] = 9
month['oct'] = 10
month['nov'] = 11
month['dec'] = 12

day_of_week = {}
day_of_week['mon'] = 1
day_of_week['tue'] = 2
day_of_week['wed'] = 3
day_of_week['thu'] = 4
day_of_week['fri'] = 5

poutcome = {}
poutcome['failure'] = 0
poutcome['success'] = 1
poutcome['nonexistent'] = 2

numeric_index = [0,10,11,12,13,15,16,17,18,19]
categorical_index = [1,2,3,4,5,6,7,8,9,14]
binary = [20]

category_dict = {}
category_dict[1] = job
category_dict[2] = marital
category_dict[3] = education
category_dict[4] = default
category_dict[5] = housing
category_dict[6] = loan
category_dict[7] = contact
category_dict[8] = month
category_dict[9] = day_of_week
category_dict[14] = poutcome

#output variable
output_y = {}
output_y['no'] = 0
output_y['yes'] = 1


def mapper_normalizer(x):
	z = str(x)
	tokens = z.split(',')
	n = len(tokens)
	normal = ""
	if n != 21:
		print "ERROR: NUMBER OF ATTRIBUTES NOT EQUAL TO 21"
	
	for i in range(0,21):
		if i == 20:
			val = str(tokens[i][1:-1])
			normal_val = str(output_y[val])
			normal += normal_val
		elif i in numeric_index: #numeric
			val = str(tokens[i])
			normal += val + ','
		elif i in categorical_index: #categorical
			val = str(tokens[i][1:-1])
			ref_dict = category_dict[i]
			normal_val = ref_dict[val]
			normal_val = str(normal_val)
			normal += normal_val + ','
	
	return normal

in_full = open("../data/bank-additional-full.csv")
in_sample = open("../data/bank-additional.csv")

out_full = open("../data/bank-additional-full_normalized.csv",'w')
out_sample = open("../data/bank-additional-sample_normalized.csv",'w')

serial_start = time.time()
for line in in_full:
	norm = mapper_normalizer(line[:-1])
	norm+= '\n'
	out_full.write(norm)
serial_end = time.time()

for line in in_sample:
	norm = mapper_normalizer(line[:-1])
	norm+= '\n'
	out_sample.write(norm)

in_full.close()
out_full.close()
in_sample.close()
out_sample.close()

print "************************SERIAL CODE DONE**************************"

###################################################################################################################
#building Spark Configuration and Getiing a Spark Context and Loading Data into an RDD
conf = (SparkConf().setMaster("local[4]").setAppName("poker_hand_logistic_regression").set("spark.executor.memory", "1g"))
sc = SparkContext(conf = conf)

data_full = sc.textFile("../data/bank-additional-full.csv")
data_sample = sc.textFile("../data/bank-additional.csv")

spark_start = time.time()
mapped_data_full = data_full.map(mapper_normalizer)
mapped_data_full.saveAsTextFile("../data/bank-additional-full_spark_normalized.csv")
spark_end = time.time()

mapped_data_sample = data_sample.map(mapper_normalizer)
mapped_data_sample.saveAsTextFile("../data/bank-additional-sample_spark_normalized.csv")

print "************************SPARK CODE DONE**************************"

serial_time = serial_end - serial_start
spark_time = spark_end - spark_start

print "\n##############Efficiency################"
print("Serial Time = " + str(serial_time))
print("Spark Time = " + str(spark_time))
print "########################################"
