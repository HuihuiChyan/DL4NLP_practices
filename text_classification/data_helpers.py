import tensorflow as tf 
import re
import numpy as np
def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def batch_iter(data, batch_size, num_epoches, shuffle=True):
	batch_num_per_epoch = len(data)//batch_size+1
	for epoch in range(num_epoches):
		start_id = 0
		end_id = batch_size
		for batch in range(batch_num_per_epoch):
			if end_id > len(data):
				end_id = len(data)
			yield data[start_id:end_id]
			start_id = end_id
			end_id = end_id+batch_size
def load_data_and_labels(positive_data_file, negative_data_file):
	pos_file = open(positive_data_file,"r",encoding="UTF-8")
	neg_file = open(negative_data_file,"r",encoding="UTF-8")
	pos_text = list(pos_file.readlines())
	neg_text = list(neg_file.readlines())
	pos_text = [clean_str(line) for line in pos_text]
	neg_text = [clean_str(line) for line in neg_text]
	len_pos = len(pos_text)
	len_neg = len(neg_text)
	pos_labels = [[0,1] for _ in range(len_pos)]
	neg_labels = [[1,0] for _ in range(len_neg)]
	#print(pos_text[1])
	#print(pos_text[2])
	#print(neg_text[1])
	#print(neg_text[2])
	#print(pos_labels[:5])
	#print(neg_labels[:5])
	all_text = pos_text + neg_text
	all_labels = pos_labels + neg_labels
	return (all_text, all_labels)