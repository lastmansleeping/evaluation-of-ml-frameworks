full = '../data/higgs.csv'
test = '../data/higgs-test.csv'
train = '../data/higgs-train.csv'

train_file = open(train, 'w')
test_file = open(test, 'w')
i = 11000000

with open(full, 'r') as infile:
    for line in infile:
        if (i%50000 == 0):
            print i
        if (i <= 10500000):
            train_file.write(line)
        else:
            test_file.write(line)
        i = i-1;

train_file.close()
test_file.close()