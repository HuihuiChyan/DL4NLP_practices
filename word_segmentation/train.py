from utils import *
from model import NN_MODEL
import tensorflow as tf

vocab = read_vocab()
config = dict()
config['maxlen'] = 32
config['lr'] = 1e-3
config['vocab_size'] = len(vocab)
config['dropout'] = 0.8
config['classes'] = 5
config['hidden_size'] = 100
config['embedding_size'] = 200

model = NN_MODEL(config)

txts, tags = read_file()
txts_idxs = np.array([sent2idx(sent, vocab) for sent in sents], dtype=np.int32)
tags_idxs = np.array([tag2idx(tag) for tag in tags], dtype=np.int32)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(10):
	train_data = generate_batch(txts_idxs, tags_idxs, batch_size=16)
	accs = []
	losses = []
	step = 0
	for batch_sents, batch_labels in train_data:
		loss_val, acc_val, _ = sess.run([model.loss, model.accuracy, model.train],\
		 feed_dict={model.input_sent:batch_sents, model.input_label:batch_labels})
		accs.append(acc_val)
		losses.append(loss_val)
		if step % 100 == 0:
			print("step: "+str(step+1)+", acc: "+str(acc_val))
		step += 1