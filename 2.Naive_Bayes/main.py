import sys
import re
import os
import math
import time


def ReadAllMail(file_dir, st_addr):
    file_list = os.listdir(file_dir)
    read_dict = {}
    with open(st_addr) as st:
        stop_words = st.read()
        stop_words = stop_words.replace('\n', ' ').lower().split(' ')
    for f in file_list:
        text = open(file_dir + f, errors='ignore').read().lower()
        for char in '!@#$%^&*()+,-/:;<=>?@[]_`~{|}\"\\\n':
            text.replace(char, ' ')
        for char in '0123456789':
            text = re.sub(char, ' ', text)
        arr = re.findall(r"\w+", text)
        for word in arr:
            if word not in stop_words:
                read_dict[word] = read_dict.get(word, 0) + 1
    read_list = list(read_dict.items())
    read_list.sort(key=lambda x: x[1], reverse=True)
    read_stat = []
    for item in read_list:
        read_stat.append([item[0], item[1]])
    return read_stat


def ReadSingleMail(file_addr, st_addr):
    read_dict = {}
    with open(st_addr) as st:
        stop_words = st.read()
        stop_words = stop_words.replace('\n', ' ').lower().split(' ')
    text = open(file_addr, errors='ignore').read().lower()
    for char in '!@#$%^&*()+,-/:;<=>?@[]_`~{|}\"\\\n':
        text.replace(char, ' ')
    for char in '0123456789':
        text = re.sub(char, ' ', text)
    arr = re.findall(r"\w+", text)
    for word in arr:
        if word not in stop_words:
            read_dict[word] = read_dict.get(word, 0) + 1
    read_list = list(read_dict.items())
    read_list.sort(key=lambda x: x[1], reverse=True)
    read_stat = []
    for item in read_list:
        read_stat.append([item[0], item[1]])
    return read_stat


def Corelate(spam_stat, ham_stat):
    spam_dict = {}
    ham_dict = {}
    for item in spam_stat:
        spam_dict[item[0]] = item[1]
    for item in ham_stat:
        ham_dict[item[0]] = item[1]
    spam_list_new = []
    ham_list_new = []
    for item in spam_stat:
        if item[0] in ham_dict:
            spam_list_new.append(item)
    for item in ham_stat:
        if item[0] in spam_dict:
            ham_list_new.append(item)
    return spam_list_new, ham_list_new


def Predict(file_addr, st_addr, spam_dict, ham_dict, spam_ratio):
    read_stat = ReadSingleMail(file_addr, st_addr)
    possibility = [math.log(1-spam_ratio), math.log(spam_ratio)]
    for item in read_stat:
        if (item[0] in ham_dict) & (item[0] in spam_dict):
            possibility[0] += item[1] * ham_dict[item[0]]
            possibility[1] += item[1] * spam_dict[item[0]]
    if possibility[0] > possibility[1]:
        return 0
    else:
        return 1


start = time.clock()
ham_dir = "./data/enron/train/ham/"
spam_dir = "./data/enron/train/spam/"
ham_test_dir = "./data/enron/test/ham/"
spam_test_dir = "./data/enron/test/spam/"
stop_words_addr = "./data/english_stopwords.txt"
spam_num = len(os.listdir(spam_dir))
ham_num = len(os.listdir(ham_dir))
spam_test_num = len(os.listdir(spam_test_dir))
ham_test_num = len(os.listdir(ham_test_dir))
spam_ratio = spam_num / (spam_num + ham_num)
# Preliminaries
spam_stat = ReadAllMail(spam_dir, stop_words_addr)
ham_stat = ReadAllMail(ham_dir, stop_words_addr)
spam_stat, ham_stat = Corelate(spam_stat, ham_stat)
# Read and flush concatenated files
spam_word_num = 0
ham_word_num = 0
for item in spam_stat:
    spam_word_num += item[1]
for item in ham_stat:
    ham_word_num += item[1]
# Count
for i in range(len(spam_stat)):
    spam_stat[i][1] = math.log(spam_stat[i][1] / spam_word_num)
for i in range(len(ham_stat)):
    ham_stat[i][1] = math.log(ham_stat[i][1] / ham_word_num)
# Get logarithm values of statistical results
spam_dict = {}
ham_dict = {}
for item in spam_stat:
    spam_dict[item[0]] = item[1]
for item in ham_stat:
    ham_dict[item[0]] = item[1]
# Transform to dictionary for column matching
correct_count = 0
for f in os.listdir(spam_test_dir):
    result = Predict(spam_test_dir + f, stop_words_addr,
                     spam_dict, ham_dict, spam_ratio)
    if result == 1:
        correct_count += 1
for f in os.listdir(ham_test_dir):
    result = Predict(ham_test_dir + f, stop_words_addr,
                     spam_dict, ham_dict, spam_ratio)
    if result == 0:
        correct_count += 1
accuracy = correct_count / (ham_test_num + spam_test_num)
period = time.clock() - start
# Predict & Evaluate
print("accuracy = %.6f" % accuracy)
print("time cost: ", period)
for i in range(10):
    word, count = ham_stat[i]
    print('{0:<15}{1:>5}'.format(word, count))
print('\n')
for i in range(10):
    word, count = spam_stat[i]
    print('{0:<15}{1:>5}'.format(word, count))
# Show top-10 features
