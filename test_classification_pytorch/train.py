import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from collections import defaultdict

EMBED_SIZE = 16
MAX_LEN = 50
BATCH_SIZE = 100
LEARNING_RATE = 0.01
EPOCH_NUM = 100
RANDOM_SEED = 10
def zero():
	return 0
def pad_sents(sents):
	new_sents = []
	for sent in sents:
		if len(sent)<=MAX_LEN:
			new_sent = sent + ['PAD' for i in range(MAX_LEN-len(sent))]
		elif len(sent)>MAX_LEN:
			new_sent = sent[:MAX_LEN]
		new_sents.append(new_sent)
	return(new_sents)

with open("./data/pos.train", "r", encoding="utf-8") as pos_train_f,\
	open("./data/neg.train", "r", encoding="utf-8") as neg_train_f,\
	open("./data/pos.dev", "r", encoding="utf-8") as pos_dev_f,\
	open("./data/neg.dev", "r", encoding="utf-8") as neg_dev_f,\
	open("./data/vocab.txt", "r", encoding="utf-8") as vocab_f:

	pos_train_sents = [sent.strip().split() for sent in pos_train_f.readlines()]
	neg_train_sents = [sent.strip().split() for sent in neg_train_f.readlines()]
	pos_dev_sents = [sent.strip().split() for sent in pos_dev_f.readlines()]
	neg_dev_sents = [sent.strip().split() for sent in neg_dev_f.readlines()]
	pos_train_sents = pad_sents(pos_train_sents)
	neg_train_sents = pad_sents(neg_train_sents)
	pos_dev_sents = pad_sents(pos_dev_sents)
	neg_dev_sents = pad_sents(neg_dev_sents)
	print("Mark here.")

	vocab = [word.strip() for word in vocab_f.readlines()]
	vocab = ['UNK', 'PAD'] + vocab
	len_vocab = len(vocab)
	word2id = defaultdict(zero)
	id2word = {}
	for i,word in enumerate(vocab):
		word2id[word] = i
		id2word[i] = word
	pos_train_ids = [[word2id[word] for word in sent] for sent in pos_train_sents]
	neg_train_ids = [[word2id[word] for word in sent] for sent in neg_train_sents]
	pos_dev_ids = [[word2id[word] for word in sent] for sent in pos_dev_sents]
	neg_dev_ids = [[word2id[word] for word in sent] for sent in neg_dev_sents]


	#0 for positive and 1 for negative
	# pos_train_ids = [(pos_train_id, 0) for pos_train_id in pos_train_ids]
	# neg_train_ids = [(neg_train_id, 1) for neg_train_id in neg_train_ids]

	embeds = nn.Embedding(len_vocab, EMBED_SIZE)
	#print(embeds.weight)
	pos_train_tensor = torch.LongTensor(pos_train_ids)
	pos_train_embeds = embeds(pos_train_tensor)
	neg_train_tensor = torch.LongTensor(neg_train_ids)
	neg_train_embeds = embeds(neg_train_tensor)
	pos_dev_tensor = torch.LongTensor(pos_dev_ids)
	pos_dev_embeds = embeds(pos_dev_tensor)
	neg_dev_tensor = torch.LongTensor(neg_dev_ids)
	neg_dev_embeds = embeds(neg_dev_tensor)
	print("Mark here again.")

	train_embeds = torch.cat((pos_train_embeds,neg_train_embeds),0)
	dev_embeds = torch.cat((pos_dev_embeds,neg_dev_embeds),0)
	train_size = train_embeds.size()[0]
	dev_size = dev_embeds.size()[0]
	train_embeds = train_embeds.view([train_size,-1])
	dev_embeds = dev_embeds.view([dev_size,-1])

	train_tags = np.array([1 for i in range(len(pos_train_embeds))] + [0 for i in range(len(neg_train_embeds))])
	dev_tags = np.array([1 for i in range(len(pos_dev_embeds))] + [0 for i in range(len(neg_dev_embeds))])
	train_tags = torch.LongTensor(train_tags)
	dev_tags = torch.LongTensor(dev_tags)

	np.random.seed(10)
	np.random.shuffle(train_embeds)
	np.random.shuffle(train_tags)
	np.random.shuffle(dev_embeds)
	np.random.shuffle(dev_tags)
	print("Mark here again and again.")

	embeds_loader = data.DataLoader(dataset=train_embeds, batch_size=BATCH_SIZE)
	tags_loader = data.DataLoader(dataset=train_tags, batch_size=BATCH_SIZE)

class MLP(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(MLP, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.activation = nn.ReLU()
		self.fc2 = nn.Linear(hidden_size, output_size)
		# self.sequetial = torch.Sequetial(
		# 	nn.Linear(input_size, hidden_size),
		# 	nn.ReLU(),
		# 	nn.Linear(hidden_size, output_size)
		# 	)
	def forward(self, x):
		fc1_out = self.fc1(x)
		act_out = self.activation(fc1_out)
		fc2_out = self.fc2(act_out)
		return fc2_out

mlp = MLP(16*50, 50, 2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp.parameters(),
						lr=LEARNING_RATE)
print("Mark here the third time.")

for epoch in range(EPOCH_NUM):
	for i,batch in enumerate(zip(embeds_loader, tags_loader)):
		print("One round start.")
		outputs = mlp(batch[0])
		print("Model finished.")
		# print(outputs.shape)
		# print(batch[1].shape)
		# print(outputs[:3])
		# print(batch[1][:3])
		# exit()
		loss = criterion(outputs, batch[1])
		print("Loss calculated.")
		# print(loss)
		optimizer.zero_grad()
		print("Grad propagating.")
		loss.backward()
		print("Grade propagated.")
		optimizer.step()
		print("One step of optimizer.")

		print("Step: "+str(i+1)+", loss: "+str(loss))

		if (i+1)%2 == 0:
			outputs = mlp(dev_embeds)
			output_result = torch.max(outputs, dim=-1)
			ground_truth = torch.max(dev_tags, dim=-1)
			print(output_result[1].size())
			print(ground_truth[1].size())
			print(output_result)
			print(ground_truth)
			exit()
			accuracy = (output_result==ground_truth).sum()/output_result.size()[0]
			print("Step: "+str(i+1)+", accuracy: "+str(accuracy))