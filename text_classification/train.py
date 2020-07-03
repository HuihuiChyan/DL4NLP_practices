import tensorflow as tf 
import data_helpers
import os
import text_cnn
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Fuck you you just want to have sex with her.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Fuck you you do not treasure the love with her.")
tf.flags.DEFINE_integer("batch_size", 256, "Fuck you you can never meet someone like her again.")
tf.flags.DEFINE_integer("num_epoches", 100, "Fuck you you promised and then you abandon her.")
tf.flags.DEFINE_integer("num_classes", 2, "Fuck you you forget her as soon as you leave her.")
tf.flags.DEFINE_integer("embedding_size", 128, "Fuck you you find a new girl as soon as you leave her.")
tf.flags.DEFINE_integer("num_filters", 128, "Fuck you, just fuck you.")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "Fuck your asshole.")
tf.flags.DEFINE_integer("filter_size", 3, "Fuck you bitch.")
tf.flags.DEFINE_float("train_dev_split", 0.1, "Fuck.")
tf.flags.DEFINE_float("keep_prob", 0.5, "Just fuck my life.")
tf.flags.DEFINE_float("learning_rate", 0.001, "Just fuck your self.")
tf.flags.DEFINE_string("checkpoint_dir","./run","Fuck the world")
#Totally 5000 sentence for each polarity!
FLAGS = tf.flags.FLAGS

def preprocess(batched_texts, max_len, vocab):
	batch_ids = []
	i = 0
	for sentence in batched_texts:
		sentence_ids = []
		for word in sentence:
			if word in vocab:
				sentence_ids.append(vocab.index(word)+1)
			else:
				sentence_ids.append(0)
				# 0 for OOV word
		if len(sentence_ids)< max_len:
			sentence_ids.extend([0 for _ in range(max_len-len(sentence_ids))])
		elif len(sentence_ids)> max_len:
			sentence_ids = sentence_ids[:max_len]
		sentence_ids = np.array(sentence_ids)
		batch_ids.append(sentence_ids)
	batch_ids = np.array(batch_ids)
	return batch_ids

def train(texts, labels):
	np.random.seed(10)
	shuffle_indice = np.random.permutation(np.arange(len(texts)))
	texts = np.array(texts)
	texts = texts[shuffle_indice]
	texts = texts.tolist()
	labels = np.array(labels)
	labels = labels[shuffle_indice]
	labels = labels.tolist()
	split_point = int(len(texts)*(1-FLAGS.train_dev_split))
	texts = [line.strip().split() for line in texts]
	train_texts = texts[:split_point]
	dev_texts = texts[split_point:]
	train_labels = labels[:split_point]
	dev_labels = labels[split_point:]
	print("train_texts length is "+str(len(train_texts)))
	print("train_labels length is "+str(len(train_labels)))
	print("eval text is:")
	print(dev_texts[:10])
	print("eval label is:")
	print(dev_labels[:10])
	max_len = 0
	vocab = []
	for line in train_texts:
		if len(line)>max_len:
			max_len=len(line)
		for word in line:
			if word not in vocab:
				vocab.append(word)
	vocab_len = len(vocab)
	batched_labels = data_helpers.batch_iter(train_labels, FLAGS.batch_size, FLAGS.num_epoches)
	batched_texts = data_helpers.batch_iter(train_texts, FLAGS.batch_size, FLAGS.num_epoches)
	#config = tf.ConfigProto()
	#config.gpu_options.per_process_gpu_memory_fraction = 0.33
	print("Batch_iter created.")
	sess = tf.InteractiveSession()
	textcnn = text_cnn.TextCNN(
		sequence_length = max_len,
		num_classes = FLAGS.num_classes,
		vocab_size = vocab_len+1,
		embedding_size = FLAGS.embedding_size,
		filter_size = FLAGS.filter_size,
		num_filters = FLAGS.num_filters,
		l2_reg_lambda = FLAGS.l2_reg_lambda,
		vocab = vocab,
		max_len = max_len)
	optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
	saver = tf.train.Saver(max_to_keep=3)
	train_step = optimizer.minimize(textcnn.loss)
	global_step = 0
	dev_texts = preprocess(dev_texts, max_len, vocab)
	#print(dev_texts)
	#print(dev_labels)
	def train_batch(input_x, input_y):
		_, accuracy, loss = sess.run((train_step,textcnn.accuracy,textcnn.loss), 
								feed_dict = {textcnn.input_x: input_x,
											textcnn.input_y: input_y,
											textcnn.dropout_keep_prob: FLAGS.keep_prob})
		print("For "+str(global_step)+", the accuracy is "+str(accuracy)+", the loss is "+str(loss))
	def dev():
		_, accuracy, result = sess.run((train_step,textcnn.accuracy,textcnn.predict), 
								feed_dict = {textcnn.input_x: dev_texts,
											textcnn.input_y: dev_labels,
											textcnn.dropout_keep_prob: 1})
		print("One evaluation finished and the accuracy is "+ str(accuracy))
		print("My predict is:")
		print(result[:10])
	sess.run(tf.global_variables_initializer())
	for batch in zip(batched_texts, batched_labels):
		# zip two generator would create a new generator
		# then every time calling the zipped generator, it would generate a tuple
		# with the first element generated from the first generator, and the second the same
		batch_ids = preprocess(batch[0], max_len, vocab)
		array_labels = np.array(batch[1])
		#print(batch_ids)
		#print(array_labels)
		train_batch(batch_ids, array_labels)
		#batch_data = batch_data.transpose(0, 1)
		global_step += 1
		if global_step%20 == 0:
			dev()
			save_path = saver.save(sess, os.path.join(FLAGS.checkpoint_dir,"model.ckpt"), global_step = global_step)
			print("Model saved in "+str(FLAGS.checkpoint_dir))

def main(argv=None):
	texts, labels = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
	train(texts, labels)

if __name__ == '__main__':
	tf.app.run()
