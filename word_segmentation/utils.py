#encoding=utf-8
import numpy as np
taglist = {'B':1, 'M':2, 'E':3, 'S':4, 'X':0}
def read_vocab():
	path = './vocab.txt'
	data = []
	vocab = dict()
	with open(path, 'r', encoding="utf-8") as fr:
		data = fr.read().split('\n')
	vocab['[PAD]'] = 0
	for word in data:
		vocab[word.strip()] = len(vocab)
	return vocab

def read_file():
	path = './msr_training.unicode'
	data = []
	with open(path, 'r', encoding="utf-8") as fr:
		data = fr.read().split('\n')[:-1]
	sents = []
	tags = []
	for line in data:
		temp = line.strip().split('\t')
		sents.append(temp[0])
		tags.append(temp[1])
	return sents,tags

def sent2idx(sent, vocab, maxlen=32):
	idxs = np.zeros(maxlen, dtype=np.int32)
	for index,char in enumerate(sent):
		idxs[index] = vocab.get(char, 1)
	return idxs

def tag2idx(tags, maxlen=32):
	idxs = np.zeros(maxlen, dtype=np.int32)
	for index, char in enumerate(tags):
		idxs[index] = taglist.get(char, 0)
	return idxs

def generate_batch(txts_idxs, tags_idxs, batch_size=32, shuffle=True):
	if shuffle:
		permutation = np.random.permutation(len(txts_idxs))
		txts_idxs = txts_idxs[permutation]
		tags_idxs = tags_idxs[permutation]
	batch_nums = (len(txts_idxs)-1)//batch_size+1
	for i in range(batch_nums):
		start = i*batch_size
		end = min((i+1)*batch_size, len(txts_idxs))
		yield np.array(txts_idxs[start:end], dtype=np.int32),\
			np.array(tags_idxs[start:end], dtype=np.int32)

sents, tags = read_file()
vocab = read_vocab()
# print(sents[0]+'\t'+str(sent2idx(sents[0], vocab)))
# print(tags[0]+'\t'+str(tag2idx(tags[0])))
txts_idxs = np.array([sent2idx(sent, vocab) for sent in sents], dtype=np.int32)
tags_idxs = np.array([tag2idx(tag) for tag in tags], dtype=np.int32)
	
# print(np.shape(txts_idxs))
# print(np.shape(tags_idxs))