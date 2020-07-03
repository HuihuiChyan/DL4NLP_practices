import tensorflow as tf
from preprocess import *
import pickle
tf.flags.DEFINE_integer("batch_size", 128, "The size of each batch.")
tf.flags.DEFINE_integer("num_epoch", 100, "Number epoches.")
tf.flags.DEFINE_boolean("ja2ch", False, "Whether to translate Japanese into Chinese.")
tf.flags.DEFINE_integer('eval_freq', 10, 'Evaluate after certain steps.')
tf.flags.DEFINE_integer('save_freq', 10, 'Save after certain steps.')
FLAGS = tf.flags.FLAGS
with open("./data/ch_vocab.txt", "rb") as ch_vocab_f,\
	open("./data/ja_vocab.txt", "rb") as ja_vocab_f:
	ch_vocab = pickle.loads(ch_vocab_f.read())
	ja_vocab = pickle.loads(ja_vocab_f.read())
with open("./data/ja_train.txt", "rb") as ja_train_f,\
	open("./data/ch_train.txt", "rb") as ch_train_f,\
	open("./data/ja_eval.txt", "rb") as ja_eval_f,\
	open("./data/ch_eval.txt", "rb") as ch_eval_f:
	ja_train = pickle.loads(ja_train_f.read())
	ch_train = pickle.loads(ch_train_f.read())
	ja_eval = pickle.loads(ja_eval_f.read())
	ch_eval = pickle.loads(ch_eval_f.read())
ja_iter = batch_iter(ja_train, FLAGS.batch_size, FLAGS.num_epoch)
ch_iter = batch_iter(ch_train, FLAGS.batch_size, FLAGS.num_epoch)
if FLAGS.ja2ch == True:
	en_iter = ja_iter
	en_eval = ja_eval
	en_vocab = ja_vocab
	de_iter = ch_iter
	de_eval = ch_eval
	de_vocab = ch_vocab
else:
	en_iter = ch_iter
	en_eval = ch_eval
	en_vocab = ch_vocab
	de_iter = ja_iter
	de_eval = ja_eval
	de_vocab = ja_vocab
en_indexer = {v:k for k,v in en_vocab.items()}
de_indexer = {v:k for k,v in de_vocab.items()}
for epoch in range(FLAGS.num_epoch):
	for train_batch in zip(en_iter, de_iter):
		with tf.Session() as sess:
			ckpt = tf.train.get_checkpoint_state('./ckpt') 
			saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta')
			saver.restore(sess, ckpt.model_checkpoint_path)
			graph = tf.get_default_graph()
			global_step = graph.get_tensor_by_name("transformer/final_part/global_step:0")
			train_step = graph.get_tensor_by_name("transformer/final_part/train_step:0")
			loss = graph.get_tensor_by_name("transformer/final_part/loss:0")
			input_en = graph.get_tensor_by_name("transformer/embedding_part/input_en:0")
			input_de = graph.get_tensor_by_name("transformer/embedding_part/input_de:0")
			global_step_value, _, loss_result = sess.run((global_step, train_step, loss),\
			 feed_dict={input_en: train_batch[0], input_de: train_batch[1]})
			print("After %d steps, the loss is %d." % (global_step_value, loss_result))
			# summary_writer.close()
			# exit()
			if global_step_value%FLAGS.save_freq==0:
				saver.save(sess, "./ckpt/my_model")

			if global_step_value%FLAGS.eval_freq==0:
				prediction_result = sess.run(prediction, feed_dict={input_en: en_eval, input_de: de_eval})
				# print("prediction result:"+str(prediction[0]))
				# bleu_score = bleu(prediction, de_eval)
				translate_ori = [[en_indexer[char] for char in batch] for batch in en_eval]
				translate_obj = [[de_indexer[char] for char in batch] for batch in de_eval]
				translate_resl = [[de_indexer[char] for char in batch] for batch in prediction_result]
				print("The original sentence:" + str(translate_ori[0]))
				print("The objective sentence:" + str(translate_obj[0]))
				print("The result sentence:" + str(translate_resl[0]))