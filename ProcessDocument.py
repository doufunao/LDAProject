import os
import re
import Lda
import json
import jieba
import math
import cmath
import itertools
import numpy as np

def __if_special_char(string):
	chinese_pa = re.compile(r'([\u4e00-\u9fa5]{2,}?)')
	if chinese_pa.search(string):
		return False
	else:
		return True

def get_stop_words(file_name):
	current_pwd = os.getcwd()
	pwd = '/'.join([current_pwd, file_name])
	stopwords = set()
	f = open(pwd, 'r')
	for line in f:
		stopwords.add(line.strip())
	f.close()
	return stopwords

###
def get_doc_pwd_list(dir_pwd):
	doc_pwd_list = []
	doc_name_list = []
	for d in os.listdir(dir_pwd):
		path = '/'.join([dir_pwd, d])
		doc_pwd_list.append(path)
		doc_name_list.append(d)
	return doc_pwd_list, doc_name_list

###
def get_doc_str(doc_pwd):
	f = open(doc_pwd, 'r')
	doc_str = f.read()
	f.close()
	return doc_str

def get_docs_list(dir_pwd):
	doc_pwd_list, doc_name_list = get_doc_pwd_list(dir_pwd)
	doc_str_list = []
	for pwd in doc_pwd_list:
		doc_str_list.append(get_doc_str(pwd))
	return doc_str_list

def get_docs_and_filename_list(dir_pwd):
	doc_pwd_list, doc_name_list = get_doc_pwd_list(dir_pwd)
	doc_str_list = []
	for pwd in doc_pwd_list:
		doc_str_list.append(get_doc_str(pwd))
	return doc_str_list, doc_name_list

###
def del_stop_words(word_list, file_name):
	stopwords = get_stop_words(file_name)
	new_list = []
	for word in word_list:
		if word not in stopwords and not __if_special_char(word):
			new_list.append(word)
	return new_list

def docs_filter(doc_str_list):
	doc_word_list = []
	for doc in doc_str_list:
		word_list = list(jieba.cut(doc, False))
		new_list = del_stop_words(word_list, 'stopwords')
		doc_word_list.append(new_list)
	return doc_word_list

###
def get_vocabulary(dir_pwd):
	doc_str_list = get_docs_list(dir_pwd)
	vocabulary_str = '\n'.join(doc_str_list)
	voc_list = list(set(jieba.cut(vocabulary_str, False)))
	voc_list_n = del_stop_words(voc_list, 'stopwords')
	return voc_list_n

def vocabulary_to_map(voc_list):
	voc_map_list = []
	for v in voc_list:
		voc_map_list.append([ v, voc_list.index(v)])
	return voc_map_list

def map_to_vocabulary(voc_list, map_id):
	voc_map_list = vocabulary_to_map(voc_list)


def get_word_count(word_list):
	word_set = set(word_list)
	word_count_list = []
	for word in word_set:
		d = {}
		d[word] = word_list.count(word)
		word_count_list.append(d)
		del d
	return word_count

#util function based on numpy
def construct_zeros_mat(rows_nums, cols_nums):
	return np.zeros((rows_nums, cols_nums))

#main function
def LDA_main(docs_dir_name):
	current_pwd = os.getcwd()
	dir_pwd = '/'.join([current_pwd, docs_dir_name])
	documents, doc_name_list = get_docs_and_filename_list(dir_pwd)
	vocabulary = get_vocabulary(dir_pwd)
	words = docs_filter(documents)
	topicNums=4
	beta=0.01
	iteration=50
	saveStep=10
	beginSaveIters=10
	# for topicNums in range(2,21):
	obj = Lda.Lda(vocabulary, words,topicNums = topicNums, beta = beta, iteration = iteration, saveStep = saveStep, beginSaveIters = beginSaveIters)
	obj.initialize()
	obj.inferenceModel()
	theta = obj.theta
	# print(theta)
	# print(doc_name_list)
	# print(topicNums)
	kl_mat = construct_zeros_mat(len(theta), len(theta))
	cos_mat = construct_zeros_mat(len(theta), len(theta))
	for combi in itertools.combinations(range(len(theta)), 2):
		a = combi[0]
		b = combi[1]
	# 	print(doc_name_list[a] + ',' + doc_name_list[b])
	# 	print(cos_dist(theta[a], theta[b]))
	# 	print(float(KL(theta[a], theta[b])))
		kl_mat[a, b] = KL(theta[a], theta[b]).real
		cos_mat[a, b] = cos_dist(theta[a], theta[b])
	kl_list = kl_mat.astype(float)
	cos_list = cos_mat.astype(float)
	kl_list.dump('kl_list.npy')
	cos_list.dump('cos_list.npy')
	doc_name_map = dict()
	for name in doc_name_list:
		doc_name_map[doc_name_list.index(name)] = name
	with open('doc_name_map', 'w') as fw:
		fw.write(json.JSONEncoder().encode(doc_name_map))

def test_stop_words_func():
	dir_name = 'heike_baike'
	current_pwd = os.getcwd()
	dir_pwd = '/'.join([current_pwd, dir_name])
	f = open('voc', 'wb')
	voc = get_vocabulary(dir_pwd)
	stop = get_stop_words('stopwords')
	for x in voc:
		f.write(x.encode('utf-8'))
		f.write('\n'.encode('utf-8'))
	f.close()

def KL(a, b):
	if len(a) != len(b):
		return None
	result = 0
	for x, y in zip(a, b):
		result += x * cmath.log(x/y)
	return result

def cos_dist(a, b):
    if len(a) != len(b):
        return None
    part_up = 0.0
    a_sq = 0.0
    b_sq = 0.0
    for a1, b1 in zip(a,b):
        part_up += a1*b1
        a_sq += a1**2
        b_sq += b1**2
    part_down = math.sqrt(a_sq*b_sq)
    if part_down == 0.0:
        return None
    else:
        return part_up / part_down

if __name__ == '__main__':
	dir_name = 'back_docs'
	LDA_main(dir_name)
	# test_stop_words_func()
