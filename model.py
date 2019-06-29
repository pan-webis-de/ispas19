import datetime
import time
from math import sqrt, ceil


import numpy as np
from sklearn.svm import NuSVC

import rw_operations


class NuSVClassifier(object):
    def __init__(self, language, test_path, miu=0.01, load_kernel=False):
        self.miu = miu
        self.classifier = NuSVC(self.miu, kernel='precomputed')

        self.language = language
        self.kernel_sizes = [3]
        self.kernel_path = './data/processed/' + language + '/kernel/'

        self.train_path = '/media/training-datasets/author-profiling/pan19-author-profiling-training-dataset-2019-02-18/' + \
                          language + '/'
        self.truth_file_path = './truth-' + language + '.txt'
        self.train_data_info = rw_operations.get_train_data_info(self.truth_file_path)
        self.train_labels = DataInfo.extract_labels(self.train_data_info)

        self.dataset_test_path = test_path
        self.test_data_info = rw_operations.get_test_data_info(self.language, self.dataset_test_path)
        self.dataset_size = len(self.train_data_info) + len(self.test_data_info)

        self.cache = np.empty([self.dataset_size, ], dtype=object)
        self.kernel = np.empty([self.dataset_size, self.dataset_size], dtype=float)
        self.computed_kernel = np.zeros([self.dataset_size, self.dataset_size], dtype=float)

        print('Kernel shape:', self.kernel.shape)
        print('Computed kernel shape:', self.computed_kernel.shape)

        self.compute_kernel(load_kernel)

    def __init_p_grams(self, xml_files, p_gram=3):
        index = 0
        for file in xml_files:
            tweets = rw_operations.get_tweets(file)
            self.cache[index] = rw_operations.tweets_to_p_grams(tweets, p_gram)

            index += 1

    def __normalize_kernel(self):
        for i in range(self.dataset_size):
            for j in range(self.dataset_size):
                if i != j:
                    self.kernel[i][j] /= sqrt(self.kernel[i][i] * self.kernel[j][j] + 1)

        for i in range(self.dataset_size):
            self.kernel[i][i] = 1

    def compute_kernel(self, load_kernel=False):
        if load_kernel is False:
            for size in self.kernel_sizes:
                self.create_kernel(size)
                self.computed_kernel += self.kernel
        elif load_kernel is True:
            for size in self.kernel_sizes:
                self.load_kernel(size)
                print('Loaded kernel shape:', self.kernel.shape)
                self.computed_kernel += self.kernel

        self.computed_kernel /= len(self.kernel_sizes)

    def create_kernel(self, p_gram=1):
        start_time = time.time()

        xml_files_train = [str(self.train_path + entry.author_id + '.xml') for entry in self.train_data_info]
        xml_files_test = [str(self.dataset_test_path + self.language + '/' + entry.author_id + '.xml') for entry in self.test_data_info]
        xml_files = xml_files_train + xml_files_test

        self.__init_p_grams(xml_files, p_gram)

        progress = set()
        print('Start creating kernel with p_gram: ', p_gram)
        for i in range(self.dataset_size):
            for j in range(i, self.dataset_size):
                # compute_intersection_kernel should be pass as param method
                self.kernel[i][j] = Kernel.compute_intersection_kernel(self.cache[i], self.cache[j])
                self.kernel[j][i] = self.kernel[i][j]

            percent = int(ceil((i / self.dataset_size) * 100.0))
            if percent > 0 and percent % 5 == 0 and percent not in progress:
                print('Created', percent, '% from kernel in', datetime.timedelta(seconds=time.time() - start_time))
                progress.add(percent)

        self.__normalize_kernel()

        file_name = 'kernel_' + str(p_gram) + '.txt'
        rw_operations.save_kernel(self.kernel_path + file_name, self.kernel)

    def load_kernel(self, p_gram=1):
        file_name = 'kernel_' + str(p_gram) + '.txt'
        self.kernel = np.loadtxt(self.kernel_path + file_name)

    def fit(self):
        train_size = len(self.train_data_info)
        self.classifier.fit(self.computed_kernel[0:train_size, 0:train_size], self.train_labels)

    def predict(self):
        train_size = len(self.train_data_info)
        test_size = len(self.test_data_info)
        raw_predictions = self.classifier.predict(self.computed_kernel[train_size:, 0:train_size])

        predictions = []
        for index in range(test_size):
            predictions.append(Prediction(self.test_data_info[index].author_id, self.language, raw_predictions[index]))

        return predictions


class Prediction(object):
    def __init__(self, author_id, language, value):
        self.id = author_id
        self.lang = language
        self.__set_type_gender(value)

    def __set_type_gender(self, value):
        if value == 0:
            self.type = 'bot'
            self.gender = 'bot'
        elif value == 1:
            self.type = 'human'
            self.gender = 'male'
        elif value == 2:
            self.type = 'human'
            self.gender = 'female'


class DataInfo(object):
    def __init__(self, token):
        if isinstance(token, list):
            self.author_id = token[0]
            self.type = token[1]
            self.gender = token[2]
        else:
            self.author_id = token

    @staticmethod
    def extract_labels(data_info):
        genders = [info.gender for info in data_info]

        labels = []
        for gender in genders:
            labels.append(['bot', 'male', 'female'].index(gender.strip()))

        return labels


class Kernel(object):
    @staticmethod
    def compute_intersection_kernel(s, t):
        ret = 0

        if len(s) < len(t):
            for pair in s.keys():
                a = s[pair]
                b = t[pair]
                if a < b:
                    ret += a / b
                else:
                    ret += b / a
        else:
            for pair in t.keys():
                a = s[pair]
                b = t[pair]
                if a < b:
                    ret += a / b
                else:
                    ret += b / a

        return ret
