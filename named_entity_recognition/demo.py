import tensorflow as tf
from data import sents2id
from data import pad_sequence
from data import id2label
import pickle
def demo_one(args):
	print("Please input your Chinese sentence:")
	sentence = input()
	vocab_file = args.vocab_file
	model_path = args.model_path
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
	result = graph.get_operation_by_name("accuracy/result").outputs[0]
	right_ner = graph.get_operation_by_name("accuracy/right_ner").outputs[0]
	recog_ner = graph.get_operation_by_name("accuracy/recog_ner").outputs[0]
	all_ner = graph.get_operation_by_name("accuracy/all_ner").outputs[0]
	shape = graph.get_operation_by_name("accuracy/shape").outputs[0]

	sentence = list(sentence)
	ori_sentence = sentence
	sent_len = len(sentence)
	sent_labels = [[0 for _ in range(sent_len)] for _ in range(256)]
	seq_len = [sent_len]
	seq_len.extend([0 for _ in range(255)])
	sentence = sents2id([sentence], vocab)
	sentence.extend([[1 for _ in range(sent_len)] for _ in range(255)])

	#print(sentence)
	#print(sent_labels)
	#print(seq_len)

	result = sess.run(result, feed_dict={input_sents:sentence,
											input_labels:sent_labels,
											sequence_lengths:seq_len})
	result_labels = id2label([result])
	result_labels = result_labels[0][:sent_len]
	for i in range(sent_len):
		print(str(ori_sentence[i])+"  "+str(result_labels[i]))