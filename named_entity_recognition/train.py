import tensorflow as tf
import numpy as np
import pickle
import random
import argparse
import re
import os
from model import BiLSTM_CRF
from data import batch_generate
from data import sents2id
from data import labels2id
from data import pad_sequence
from demo import demo_one


def build_vocab(train_sents, min_count, vocab_file):
	char_count = {}
	for sent in train_sents:
		for char in sent:
			if re.match(r'[a-zA-Z0-9]+', char):
				continue
			elif char not in char_count.keys():
				char_count[char] = 0
			else:
				char_count[char] += 1
	print("Originally %d characters." % (len(list(char_count.keys()))))
	for char in list(char_count.keys()):
		if char_count[char]<=min_count:
			del char_count[char]
	print("After delete %d characters." % (len(list(char_count.keys()))))
	vocab = {"UNK":0,"PAD":1}
	vocab_len = 1
	for char in char_count.keys():
		vocab_len += 1
		vocab[char] = vocab_len
	print("Totally %d characters stored in vocabulary." % (len(list(vocab.keys()))))
	with open(vocab_file, "wb") as vocab_f:
		pickle.dump(vocab, vocab_f)
		print("vocab.pk dumped.")
	#print("the top 10 words:")
	#print(list(vocab.keys())[:10])
	#print("the bottom 10 words:")
	#print(list(vocab.keys())[-10:])
	return vocab

def preprocess(train_data, train_eval_split, data_fold, shuffle=True):
	#with open(vocab_file, "rb", encoding="utf-8") as vocab_f:
	#	vocab = pickle.load(vocab_f)
	with open(train_data, "r", encoding="utf-8") as train_data_f:
		train_data = train_data_f.readlines()
		sents = []
		sent_labels = []
		chars = []
		labels = []
		for line in train_data:	
			if line[0] != "\n":
				chars.append(line[0])
				labels.append(''.join(line[2:-1]))
			else:
				sents.append(chars)
				sent_labels.append(labels)
				chars = []
				labels = []
		print("Totally %d sentences processed." % (len(sents)))
	if shuffle == True:
		shuffle_indices = np.random.permutation(len(sents))
		sents = np.array(sents)
		sents = sents[shuffle_indices]
		sents = sents.tolist()
		sent_labels = np.array(sent_labels)
		sent_labels = sent_labels[shuffle_indices]
		sent_labels = sent_labels.tolist()
	#print("sents and sent_labels shuffled.")
	#print("The first 10 sentences:")
	#print(sents[3:6])
	#print("The last 10 sentences:")
	#print(sents[-10:])
	#print("The first 10 labels:")
	#print(sent_labels[3:6])
	#print("The last 10 labels:")
	#print(sent_labels[-10:])
	split_point = int(len(sents)*train_eval_split)
	train_sents = sents[:split_point]
	train_sent_labels = sent_labels[:split_point]
	eval_sents = sents[split_point:]
	eval_sent_labels = sent_labels[split_point:]
	f1,f2,f3,f4 = open_file(data_fold, "wb")
	pickle.dump(train_sents, f1)
	pickle.dump(train_sent_labels, f2)
	pickle.dump(eval_sents, f3)
	pickle.dump(eval_sent_labels, f4)
	f1.close()
	f2.close()
	f3.close()
	f4.close()
	print("Train and eval data splitted.")
	return (train_sents, train_sent_labels, eval_sents, eval_sent_labels)

def train(args):
	train_data = args.train_data
	train_eval_split = args.train_eval_split
	min_count = args.min_count
	vocab_file = args.vocab_file
	max_step = args.max_step
	model_path = args.model_path
	data_fold = args.data_fold
	train_sents, train_sent_labels, eval_sents, eval_sent_labels = preprocess(train_data, train_eval_split, data_fold)
	vocab = build_vocab(train_sents, min_count, vocab_file)
	bilstm_crf = BiLSTM_CRF(args.batch_size, args.embedding_size,
								args.hidden_size, args.lr,
								len(list(vocab.keys())), tag_num=7)
	print("bilstm_crf object created.")
	sess = tf.InteractiveSession()
	saver = tf.train.Saver(max_to_keep = 3)
	sess.run(tf.global_variables_initializer())
	batched_train_sents = batch_generate(train_sents, args.batch_size)
	batched_train_labels = batch_generate(train_sent_labels, args.batch_size)
	print("Batch generator created.")
	while(1):
		for batch in zip(batched_train_sents, batched_train_labels):
			#print(batch[0][:10])
			#print(batch[1][:10])
			batch_sents, batch_labels, seq_len = batch_preprocess(batch, vocab)
			loss_value, _, global_step_value = sess.run((bilstm_crf.loss, bilstm_crf.train_step, bilstm_crf.global_step), 
				feed_dict = {bilstm_crf.input_sents: batch_sents,
								bilstm_crf.input_labels: batch_labels,
								bilstm_crf.sequence_lengths: seq_len})
			print("%d step finished." % (global_step_value))
			if (global_step_value%10 == 0) or (global_step_value == 1):
				f_score, right_ner, recog_ner, all_ner, shape = sess.run((bilstm_crf.f_score, 
					bilstm_crf.right_ner, 
					bilstm_crf.recog_ner,
					bilstm_crf.all_ner,
					bilstm_crf.shape),
					feed_dict = {bilstm_crf.input_sents: batch_sents,
							 		bilstm_crf.input_labels: batch_labels,
									bilstm_crf.sequence_lengths: seq_len})
				print("%d step, loss is %s, f_score is %s" % (global_step_value,
																str(loss_value),
																str(f_score)))
				print("length is %s, right_ner is %d, recog_ner is %d, all_ner is %d" % (str(shape),right_ner,
																recog_ner,
																all_ner))
				saver.save(sess, os.path.join(model_path, "model.ckpt"))

def batch_preprocess(batch, vocab):
	batch_sents = sents2id(batch[0], vocab)
	batch_labels = labels2id(batch[1])
	sequence_lengths, batch_sents = pad_sequence(batch_sents)
	_, batch_labels = pad_sequence(batch_labels)
	return batch_sents, batch_labels, sequence_lengths

def open_file(data_fold, mode):
	f1 = open(os.path.join(data_fold, "train_sents"), mode)
	f2 = open(os.path.join(data_fold, "train_sent_labels"), mode)
	f3 = open(os.path.join(data_fold, "eval_sents"), mode)
	f4 = open(os.path.join(data_fold, "eval_sent_labels"), mode)
	return f1,f2,f3,f4

def inference(args):
	train_data = args.train_data
	train_eval_split = args.train_eval_split
	min_count = args.min_count
	vocab_file = args.vocab_file
	max_step = args.max_step
	model_path = args.model_path
	data_fold = args.data_fold
	f1,f2,f3,f4 = open_file(data_fold, "rb")
	train_sents = pickle.load(f1)
	train_sent_labels = pickle.load(f2)
	eval_sents = pickle.load(f3)
	eval_sent_labels = pickle.load(f4)
	f1.close()
	f2.close()
	f3.close()
	f4.close()
	with open(vocab_file, "rb") as vocab_f:
		vocab = pickle.load(vocab_f)
	sess = tf.InteractiveSession()
	ckpt = tf.train.get_checkpoint_state(model_path)
	saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path+".meta")
	saver.restore(sess, ckpt.model_checkpoint_path)

	graph = tf.get_default_graph()
	input_sents = graph.get_operation_by_name("embedding_layer/input_sents").outputs[0]
	input_labels = graph.get_operation_by_name("embedding_layer/input_labels").outputs[0]
	sequence_lengths = graph.get_operation_by_name("embedding_layer/sequence_lengths").outputs[0]
	#log_likelihood = graph.get_operation_by_name("loss/log_likelihood").outputs[0]
	loss = graph.get_operation_by_name("loss/loss").outputs[0]
	train_step = graph.get_operation_by_name("loss/train_step").outputs[0]
	result = graph.get_operation_by_name("accuracy/result").outputs[0]
	f_score = graph.get_operation_by_name("accuracy/f_score").outputs[0]
	global_step = graph.get_operation_by_name("embedding_layer/global_step").outputs[0]
	right_ner = graph.get_operation_by_name("accuracy/right_ner").outputs[0]
	recog_ner = graph.get_operation_by_name("accuracy/recog_ner").outputs[0]
	all_ner = graph.get_operation_by_name("accuracy/all_ner").outputs[0]
	shape = graph.get_operation_by_name("accuracy/shape").outputs[0]
	#global_step = graph.get_operation_by_name("global_step").outputs[0]

	batched_train_sents = batch_generate(train_sents, args.batch_size)
	batched_train_labels = batch_generate(train_sent_labels, args.batch_size)
	while(1):
		for batch in zip(batched_train_sents, batched_train_labels):
			batch_sents, batch_labels, seq_len = batch_preprocess(batch, vocab)
			#print(seq_len)
			loss_value, _, global_step_value = sess.run((loss, train_step, global_step), 
				feed_dict={input_sents:batch_sents,
							input_labels:batch_labels,
							sequence_lengths:seq_len})
			print("%d step finished." % (global_step_value))
			if global_step_value%10 == 0:
				f_score_value, right_ner_value, recog_ner_value, all_ner_value, shape_value = sess.run((f_score, 
						right_ner, 
						recog_ner,
						all_ner,
						shape),
						feed_dict = {input_sents: batch_sents,
								 		input_labels: batch_labels,
										sequence_lengths: seq_len})
				print("%d step, loss is %s, f_score is %s" % (global_step_value,
																	str(loss_value),
																	str(f_score_value)))
				print("length is %s, right_ner is %d, recog_ner is %d, all_ner is %d" % (str(shape_value),right_ner_value,
																recog_ner_value,
																all_ner_value))
				save_path = saver.save(sess, os.path.join(model_path, "model.ckpt"))

def main(argv=None):
	parser = argparse.ArgumentParser()
	parser.add_argument("--batch_size", type=int, default=256)
	parser.add_argument("--embedding_size", type=int, default=128)
	parser.add_argument("--hidden_size", type=int, default=300)
	parser.add_argument("--train_data", type=str, default="./data/train_data")
	parser.add_argument("--data_fold", type=str, default="./data")
	parser.add_argument("--train_eval_split", type=float, default=0.1)
	parser.add_argument("--min_count", type=int, default=3)
	parser.add_argument("--vocab_file", type=str, default="./data/vocab_file")
	parser.add_argument("--model_path", type=str, default="./model")
	parser.add_argument("--max_step", type=int, default=10000)
	parser.add_argument("--lr", type=float, default=0.1)
	parser.add_argument("--is_demo", type=bool, default=False)
	parser.add_argument("--is_inference", type=bool, default=False)
	args = parser.parse_args()
	if args.is_demo == True:
		demo_one(args)
	elif args.is_inference == True:
		inference(args)
	else:
		train(args)
	
if __name__ == "__main__":
	tf.app.run()