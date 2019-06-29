import os
import re
import xml.etree.ElementTree as ET
from collections import Counter
from xml.dom import minidom

import model


def get_tweets(file_path):
    author_tweets = minidom.parse(file_path)
    tweets = author_tweets.getElementsByTagName('document')

    tweets_as_str = [tweet.firstChild.data for tweet in tweets]
    return tweets_as_str


def tweets_to_p_grams(tweets, p_gram=3):
    text = ' '.join(tweets)
    text = text.lower()
    text = re.sub(r'^https:\/\/.*[\r\n]*', 'secure', text)
    text = re.sub(r'^http:\/\/.*[\r\n]*', 'unsecure', text)
    text = ' '.join(text.split())

    counter = Counter([text[i:i + p_gram] for i in range(0, len(text) - p_gram + 1)])
    norm = 1 + sum(counter.values())
    for pair in counter.keys():
        counter[pair] /= norm

    return counter


def get_train_data_info(truth_file_path):
    train_data_info = []

    with open(truth_file_path, 'r') as truth_file:
        lines = truth_file.readlines()

        for line in lines:
            tokens = line.split(':::')
            train_data_info.append(model.DataInfo(tokens))

    return train_data_info


def get_test_data_info(language, dataset_test_path):
    test_data_info = []

    dataset_test_path += language + '/'
    train_files = [file for file in os.listdir(dataset_test_path) if file.endswith('.xml')]

    for file in train_files:
        data = ET.parse(dataset_test_path + file).getroot()
        lang = data.attrib['lang']
        author_id = file.split('.')[0]

        if lang == language:
            test_data_info.append(model.DataInfo(author_id))

    return test_data_info


def save_predictions(predictions, language, output_dir):
    print('Save predictions for language: ', language)
    output_dir = output_dir + language
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for prediction in predictions:
        data = ET.Element('author')
        data.set('id', prediction.id)
        data.set('lang', prediction.lang)
        data.set('type', prediction.type)
        data.set('gender', prediction.gender)

        author_file = open(output_dir + '/' + prediction.id + '.xml', 'wb')
        author_file.write(ET.tostring(data))


def save_kernel(file_path, kernel):
    kernel_size = len(kernel)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    print('Save kernel')

    with open(file_path, 'w+') as kernel_file:
        for i in range(kernel_size):
            for j in range(kernel_size):
                kernel_file.write("%1.4f " % kernel[i][j])
            kernel_file.write('\n')

    kernel_file.close()
