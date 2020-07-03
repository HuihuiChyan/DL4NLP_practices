from collections import Counter
from collections import defaultdict
import numpy as np
SPLIT_POINT = 0.95
def zero():
	return 0
with open("./data/sents.pos", "r", encoding="utf-8") as pos_f,\
	open("./data/sents.neg", "r", encoding="utf-8") as neg_f:
	pos_sents = [sent.strip() for sent in pos_f.readlines()]
	neg_sents = [sent.strip() for sent in neg_f.readlines()]
	pos_sents_train = pos_sents[:int(SPLIT_POINT*len(pos_sents))]
	pos_sents_dev = pos_sents[int(SPLIT_POINT*len(pos_sents)):]
	neg_sents_train = neg_sents[:int(SPLIT_POINT*len(neg_sents))]
	neg_sents_dev = neg_sents[int(SPLIT_POINT*len(neg_sents)):]
	pos_train_split = [sent.split() for sent in pos_sents]
	neg_train_split = [sent.split() for sent in neg_sents]
	all_sents_train = pos_train_split + neg_train_split
	all_words = []
	for sent in all_sents_train:
		for word in sent:
			all_words.append(word)
	counter = Counter(all_words)
	common_words = counter.most_common(5000)
	vocab_words = [word[0] for word in common_words]
	# word2index = defaultdict(zero)
	# for i, word in enumerate(vocab_words):
	# 	word2index[word] = i
	# pos_sents = [[word2index[word] for word in sent] for sent in pos_sents]
	# neg_sents = [[word2index[word] for word in sent] for sent in neg_sents]
	# print(pos_sents[:10])
	# print(neg_sents[:10])
with open("./data/vocab.txt", "w", encoding="utf-8") as vocab_f,\
	open("./data/pos.train", "w", encoding="utf-8") as pos_train_f,\
	open("./data/neg.train", "w", encoding="utf-8") as neg_train_f,\
	open("./data/pos.dev", "w", encoding="utf-8") as pos_dev_f,\
	open("./data/neg.dev", "w", encoding="utf-8") as neg_dev_f:
	for sent in pos_sents_train:
		pos_train_f.write(sent+"\n")
	for sent in neg_sents_train:
		neg_train_f.write(sent+'\n')
	for sent in pos_sents_dev:
		pos_dev_f.write(sent+'\n')
	for sent in neg_sents_dev:
		neg_dev_f.write(sent+'\n')
	for word in vocab_words:
		vocab_f.write(word+'\n')