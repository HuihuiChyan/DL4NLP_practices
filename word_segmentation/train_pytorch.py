import torch
import torch.nn as nn
from utils import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-embedding_size", type=int, default=200)
parser.add_argument("-hidden_size", type=int, default=100)
parser.add_argument("-num_class", type=int, default=5)
args = parser.parse_args()
VOCAB_SIZE = 20966
model = nn.Sequential(nn.Embedding(VOCAB_SIZE, args.embedding_size),
					nn.Linear(args.embedding_size, args.hidden_size),
					nn.ReLU(),
					nn.Linear(args.hidden_size, args.num_class))
criterion = torch.nn.CrossEntropyLoss(reduce=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
def train():
	sents, tags = read_file()
	vocab = read_vocab()
	txts_idxs = np.array([sent2idx(sent, vocab) for sent in sents], dtype=np.int32)
	tags_idxs = np.array([tag2idx(tag) for tag in tags], dtype=np.int32)
	batch_iterator = generate_batch(txts_idxs, tags_idxs)
	for i, (batch_sents, batch_tags) in enumerate(batch_iterator):
		# print(i)
		# print(batch_sents)
		# print(batch_tags)
		# exit()
		batch_sents = torch.from_numpy(batch_sents).long()
		batch_tags = torch.from_numpy(batch_tags).long()
		predictions = model(batch_sents)
		# print(batch_sents.shape)
		# print(predictions.shape)
		# print(batch_tags.shape)
		predictions = predictions.reshape([-1, args.num_class])
		batch_tags = batch_tags.reshape([-1])
		# print(predictions[:10])
		# print(batch_tags[:10])
		# exit()
		for name, param in model.named_parameters():
			if param.requires_grad:
				print(name)
		exit()
		loss = criterion(predictions, batch_tags)
		#print("Epoch:{}, loss:{}".format(i, loss.item()))
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if i%30 == 0:
			print("step:"+str(i))
			print("loss:")
			print(loss.item())
			print("acc:")
			print((torch.argmax(predictions, -1)==batch_tags).float().mean())
if __name__ == "__main__":
	train()