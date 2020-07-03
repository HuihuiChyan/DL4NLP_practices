import tensorflow as tf
import data_helpers
import numpy as np
import os
tf.flags.DEFINE_string("positive_file",'./data/rt-polaritydata/rt-polarity.pos',"I love you.")
tf.flags.DEFINE_string("negative_file",'./data/rt-polaritydata/rt-polarity.neg',"I love you.")
tf.flags.DEFINE_string("checkpoint_dir",'./run',"I hate you")
tf.flags.DEFINE_integer("batch_size",256,"I miss you")
tf.flags.DEFINE_integer("num_classes",2,"I beat you")
FLAGS = tf.flags.FLAGS

def preprocess(texts, max_len, vocab):
	text_ids = []
	for sentence in texts:
		sentence_ids = []
		for word in sentence:
			# print(word)
			if word in vocab:
				sentence_ids.append(vocab.index(word)+1)
				#print("这也是一句很显眼的中文告诉你终于特么查到了一个词")
			else:
				sentence_ids.append(0)
				#print("出现了！集外词！")
		if len(sentence_ids)<max_len:
			sentence_ids.extend([0 for _ in range(max_len-len(sentence_ids))])
		elif len(sentence_ids)>max_len:
			sentence_ids = sentence_ids[:max_len]
		sentence_ids = np.array(sentence_ids)
		text_ids.append(sentence_ids)
	text_ids = np.array(text_ids) 
	return text_ids

def eval(texts, labels):
	sess = tf.InteractiveSession()
	ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
	saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta')
	saver.restore(sess, ckpt.model_checkpoint_path)

	graph = tf.get_default_graph()
	input_x = graph.get_operation_by_name("input_x").outputs[0]
	input_y = graph.get_operation_by_name("input_y").outputs[0]
	keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
	accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]
	predict = graph.get_operation_by_name("accuracy/predict").outputs[0]
	vocab_op = graph.get_operation_by_name("vocab").outputs[0]
	max_len_op = graph.get_operation_by_name("max_len").outputs[0]

	vocab = sess.run(vocab_op)
	max_len = sess.run(max_len_op)
	print("我在想，突然插进来一句中文会不会很显眼？")
	print(max_len)
	print(vocab[:100])
	vocab = vocab.tolist()
	print(vocab[:100])
	for i in range(len(vocab)):
		vocab[i] = vocab[i].decode("UTF-8")
	print(vocab[:100])

	#For this part we notice:
	# we have defined two tf.constant in our model, which are vocab and max_len
	# but if we want to use them in our processing part
	# how could we fetch the value of these two constants?
	# must we fetch them from a session?

	texts = [line.strip().split() for line in texts]
	text_ids = preprocess(texts, max_len, vocab)
	print("preprocess finished!!!")

	batch_ids = data_helpers.batch_iter(text_ids, FLAGS.batch_size, 1)
	batch_texts = data_helpers.batch_iter(texts, FLAGS.batch_size, 1)
	batch_labels = data_helpers.batch_iter(labels, FLAGS.batch_size, 1)

	all_accuracy = []

	with open(os.path.join(FLAGS.checkpoint_dir,"result.txt"),"w",encoding="UTF-8") as result_f:
		for batch in zip(batch_ids, batch_labels, batch_texts):
			#print(batch[0])
			#print(batch[1])
			#print(batch[2])
			pre, acc = sess.run((predict, accuracy),
									feed_dict = {input_x:batch[0],input_y:batch[1],keep_prob:1})
			sentences = [' '.join(line) for line in batch[2]]
			all_accuracy.append(acc)
			for i in range(len(sentences)):
				result_f.write(sentences[i]+" "+"   prediction:"+str(pre[i])+"\n")
			print("One batch finished predicting.")
			print("For this batch the accuracy is "+str(acc))
	mean_accuracy = sum(all_accuracy)/len(all_accuracy)
	print("All batches finished and the accuracy is "+str(mean_accuracy))

def main(argv = None):
	texts, labels = data_helpers.load_data_and_labels(FLAGS.positive_file, FLAGS.negative_file)
	eval(texts, labels)

if __name__ == '__main__':
	tf.app.run()
