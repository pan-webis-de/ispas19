import os
import sys
import xml.etree.ElementTree as ET

from sklearn.metrics import accuracy_score

language = sys.argv[1]
pred_path = '../results/' + language + '/'
test_truth_file = '/media/training-datasets/author-profiling/pan19-author-profiling-training-2019-02-18/' + language + '/truth.txt'

test_labels = dict()
with open(test_truth_file, 'r') as file:
    lines = file.readlines()

    for line in lines:
        tokens = line.split(':::')
        test_labels.update({tokens[0]: ['bot', 'male', 'female'].index(tokens[2].strip())})

pred_labels = dict()
pred_files = os.listdir(pred_path)
for file in pred_files:
    data = ET.parse(pred_path + file).getroot()
    attributes = data.attrib
    prediction = ['bot', 'male', 'female'].index(data.attrib['gender'])
    pred_labels.update({data.attrib['id']: prediction})

y_test = []
y_pred = []
for key in test_labels.keys():
    y_test.append(test_labels[key])
    y_pred.append(pred_labels[key])

print('Test accuracy: ', accuracy_score(y_test, y_pred))
