from collections import defaultdict
def build_data(data_file, vocab, max_len=50, pad_num=0):
	lines = [line.strip().split() for line in data_file.readlines()]
	new_lines = []
	for line in lines:
		if len(line)>=max_len:
			new_line = [vocab[char] for char in line[:max_len]]
		else:
			new_line = [vocab[char] for char in line] + [pad_num for _ in range(max_len-len(line))]
		new_lines.append(new_line)
	return new_lines

def build_vocab(vocab_file):
	lines = [line.strip() for line in vocab_file.readlines()[:-1]]
	lines = ["PAD","UNK"] + lines
	def one():
		return 1
	vocab = defaultdict(one)
	for index in range(len(lines)):
		vocab[lines[index]] = index
	return vocab
def build_batch(pos_sents, neg_sents):
	
with open("./data/vocab.txt","r",encoding="utf-8") as vocab_f:
	vocab = build_vocab(vocab_f)
with open("./data/pos.dev","r",encoding="utf-8") as pos_dev:
	pos_dev_lines = build_data(pos_dev, vocab)
	print(pos_dev_lines[:5])