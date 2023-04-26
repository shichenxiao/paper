
import os
import random
path = 'E:/classify-bible/data/termEntityTagging/'
lpath = os.listdir(path)

total = []
for filename in lpath:
    with open(path+filename, 'r') as f:
        text = []
        labels = []
        for line in f.readlines():
            words, label = line.strip().split('\t')
            if words != '.':
                text.append(words)
                labels.append(label)
            else:
                assert len(text) == len(labels)
                total.append((text, labels))
                text = []
                labels = []


random.shuffle(total)

train_data = total[int(len(total)/4):]
train_data, dev_data = train_data[int(len(train_data)/4):], train_data[:int(len(train_data)/4)]
test_data = total[:int(len(total)/4)]
print(len(total))
print(len(train_data))
print(len(dev_data))
print(len(test_data))


with open(path+'train.txt', 'w') as f:
    for line in train_data:
        text, labels = line
        for i in range(len(text)):
            f.write(text[i]+'\t'+labels[i]+'\n')
        f.write('\n')

with open(path+'dev.txt', 'w') as f:
    for line in dev_data:
        text, labels = line
        for i in range(len(text)):
            f.write(text[i]+'\t'+labels[i]+'\n')
        f.write('\n')

with open(path+'test.txt', 'w') as f:
    for line in test_data:
        text, labels = line
        for i in range(len(text)):
            f.write(text[i]+'\t'+labels[i]+'\n')
        f.write('\n')


