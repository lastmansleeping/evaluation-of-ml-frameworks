from random import shuffle
full = '../data/bank-additional-full-normalized.csv'
test = '../data/bank-additional-test-normalized.csv'
train = '../data/bank-additional-train-normalized.csv'

full_file = open(full, 'r')
full_lines = full_file.readlines()
shuffle(full_lines)
train_file = open(train, 'w')
test_file = open(test, 'w')
i = 41188

for line in full_lines:
    if (i <= 4119):
        test_file.write(line)
    else:
        train_file.write(line)
    i = i-1;
        

full_file.close()
train_file.close()
test_file.close()
