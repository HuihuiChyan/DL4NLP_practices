import tensorflow as tf
import random
import numpy as np
from collections import Counter

def build_lines(input_lines, vocab, split_point, max_len):
	# np.random.seed(10)
	# input_length = len(input_lines)
	# shuffle_indices = np.random.permutation(input_length)
	# input_lines = input_lines[shuffle_indices]

		#0 for UNK word and 1 for PAD
	output_lines = []
	for line in input_lines:
		line = line.strip().split()
		output_words = [2]
		for word in line:
			if word not in vocab.keys():
				output_words.append(0)
			else:
				output_words.append(vocab[word])
		output_words.append(3)
		output_lines.append(output_words)
	output_train = output_lines[:split_point]
	output_eval = output_lines[split_point:]
	padded_train, train_lens = batch_padding(output_train, max_len)
	padded_eval, eval_lens = batch_padding(output_eval, max_len)
	return padded_train, padded_eval, train_lens, eval_lens

def build_vocab(input_lines, vocab_size):
	input_lines = np.array(input_lines)
	np.random.shuffle(input_lines)
	input_lines = input_lines.tolist()
	input_words = []
	for line in input_lines:
		input_words.extend(line.strip().split())
	counter_words = Counter(input_words)
	vocab_words = counter_words.most_common(vocab_size)
	vocab_words = [vocab_word[0] for vocab_word in vocab_words]
	vocab = {'PAD':0, 'UNK':1, 'BOS':3, 'EOS':2}
	for i in range(len(vocab_words)):
		vocab[vocab_words[i]] = i+4
	print("Vocab size: "+str(len(vocab)))
	return vocab


def batch_iter(input_lines, batch_size, num_epoch):
	batch_num = len(input_lines)//batch_size + 1
	for i in range(num_epoch):
		start_id = 0
		end_id = batch_size
		for j in range(batch_num):
			if end_id < len(input_lines):
				yield input_lines[start_id:end_id]
			else:
				yield input_lines[start_id:]
			start_id = end_id
			end_id += batch_size

def batch_padding(input_batch, max_len, padding_mark=0):
	seq_len = []
	output_batch = []
	for line in input_batch:
		seq_len.append(len(line))
		if len(line)<max_len:
			output_batch.append(line+[padding_mark for _ in range(max_len-len(line))])
		elif len(line)>max_len:
			output_batch.append(line[:max_len])
	return output_batch, seq_len