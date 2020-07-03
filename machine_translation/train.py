import pickle
import tensorflow as tf
from preprocess import *
from bleu_cal import bleu
from model import Transformer
tf.flags.DEFINE_string('input_ja', './data/ja.seg', 'Japanese file.')
tf.flags.DEFINE_string('input_ch', './data/zh.seg', 'Chinese file.')
tf.flags.DEFINE_integer('vocab_size_ja', 3000, 'Vocabulary size.')
tf.flags.DEFINE_integer('vocab_size_ch', 3000, 'Vocabulary size.')
tf.flags.DEFINE_integer('max_len_ja', 50, 'Max length of Japanese sentences.')
tf.flags.DEFINE_integer('max_len_ch', 50, 'Max length of Chinese sentences.')
tf.flags.DEFINE_float('learning_rate', 0.1, 'Learning rate.' )
tf.flags.DEFINE_float('learning_rate_decay', 0.9, 'Learning rate decay.')
tf.flags.DEFINE_integer('en_num', 2, 'Multi head number for encoder side.')
tf.flags.DEFINE_integer('de_num', 2, 'Multi head number for decoder side.')
tf.flags.DEFINE_float('train_eval_split', 0.99, 'Train evaluation split point.')
tf.flags.DEFINE_integer('num_epoch', 100, 'Number epoches.')
tf.flags.DEFINE_boolean('ja2ch', False, 'Translation direction.')
tf.flags.DEFINE_integer('hidden_size', 512, 'Hidden size for attention layers.')
tf.flags.DEFINE_integer('eval_freq', 10, 'Evaluate after certain steps.')
tf.flags.DEFINE_integer('save_freq', 10, 'Save after certain steps.')
tf.flags.DEFINE_integer('batch_size', 128, 'The size of each batch.')
tf.flags.DEFINE_integer('embedding_size', 256, 'The size of embedding')
tf.flags.DEFINE_integer('head_num', 8, 'The size of h when using multi head attention.')
FLAGS = tf.flags.FLAGS

def train(args):
	with open(FLAGS.input_ja, 'r', encoding='utf-8') as ja_f,\
		open(FLAGS.input_ch, 'r', encoding='utf-8') as ch_f:
			ja_lines = ja_f.readlines()
			ch_lines = ch_f.readlines()
			lines_len = len(ja_lines)
			split_point = int(lines_len * FLAGS.train_eval_split)
			ja_vocab = build_vocab(ja_lines, FLAGS.vocab_size_ja)
			ch_vocab = build_vocab(ch_lines, FLAGS.vocab_size_ch)
			with open("./data/ch_vocab.txt", "wb") as ch_vocab_f,\
				open("./data/ja_vocab.txt", "wb") as ja_vocab_f:
				ch_vocab_f.write(pickle.dumps(ch_vocab))
				ja_vocab_f.write(pickle.dumps(ja_vocab))
			ja_train, ja_eval, ja_train_lens, ja_eval_lens = build_lines(ja_lines, ja_vocab, split_point, FLAGS.max_len_ja)
			ch_train, ch_eval, ch_train_lens, ch_eval_lens = build_lines(ch_lines, ch_vocab, split_point, FLAGS.max_len_ch)
			with open("./data/ja_train.txt", "wb") as ja_train_f,\
				open("./data/ch_train.txt", "wb") as ch_train_f,\
				open("./data/ja_eval.txt", "wb") as ja_eval_f,\
				open("./data/ch_eval.txt", "wb") as ch_eval_f:
				ja_train_f.write(pickle.dumps(ja_train))
				ja_eval_f.write(pickle.dumps(ja_eval))
				ch_train_f.write(pickle.dumps(ch_train))
				ch_eval_f.write(pickle.dumps(ch_eval))
			# print(ja_train)
			# exit()
			ja_iter = batch_iter(ja_train, FLAGS.batch_size, FLAGS.num_epoch)
			ch_iter = batch_iter(ch_train, FLAGS.batch_size, FLAGS.num_epoch)
	if FLAGS.ja2ch == True:
		en_iter = ja_iter
		en_eval = ja_eval
		en_vocab = ja_vocab
		de_iter = ch_iter
		de_eval = ch_eval
		de_vocab = ch_vocab
		with tf.name_scope("transformer"):
			transformer = Transformer(FLAGS.vocab_size_ja, FLAGS.vocab_size_ch,\
				FLAGS.max_len_ja, FLAGS.max_len_ch, FLAGS.embedding_size, FLAGS.head_num,\
				FLAGS.hidden_size, FLAGS.en_num, FLAGS.de_num, FLAGS.learning_rate)
			#svocab_size_en, vocab_size_de, max_len_en, max_len_de,
		 #embedding_size, h, c_size, en_num, de_num, lr
		print("Now translating Japanese to Chinese")
	else:
		en_iter = ch_iter
		en_eval = ch_eval
		en_vocab = ch_vocab
		de_iter = ja_iter
		de_eval = ja_eval
		de_vocab = ja_vocab
		with tf.name_scope("transformer"):
			transformer = Transformer(FLAGS.vocab_size_ch, FLAGS.vocab_size_ja,\
				FLAGS.max_len_ch, FLAGS.max_len_ja, FLAGS.embedding_size, FLAGS.head_num,\
				FLAGS.hidden_size, FLAGS.en_num, FLAGS.de_num, FLAGS.learning_rate)
		print("Now translating Chinese to Japanese")
	# with open("./data/ch_vocab.txt", "rb") as ch_vocab_f,\
	# 	open("./data/ja_vocab.txt", "rb") as ja_vocab_f:
	# 	ch_vocab = pickle.loads(ch_vocab_f.read())
	# 	ja_vocab = pickle.loads(ja_vocab_f.read())
	# print(ch_vocab)
	# print(ja_vocab)
	# exit()

	print("The whole model created. Now start the session.")

	sess = tf.InteractiveSession()
	sess.run(tf.initializers.global_variables())
	saver = tf.train.Saver()
	# summary_writer = tf.summary.FileWriter("./result", sess.graph)
	en_indexer = {v:k for k,v in en_vocab.items()}
	de_indexer = {v:k for k,v in de_vocab.items()}
	for epoch in range(FLAGS.num_epoch):
		for train_batch in zip(en_iter, de_iter):
			# translate_ori = [[en_indexer[char] for char in sent] for sent in train_batch[0]]
			# translate_obj = [[de_indexer[char] for char in sent] for sent in train_batch[1]]
			# print(translate_ori[:10])
			# print(translate_obj[:10])
			# exit()
			global_step_value, _, loss = sess.run((transformer.global_step, transformer.train_step, transformer.loss),\
				feed_dict={transformer.input_en: train_batch[0], transformer.input_de: train_batch[1]})
			print("After %d steps, the loss is %d." % (global_step_value, loss))
			# summary_writer.close()
			# exit()
			if global_step_value%FLAGS.save_freq==0:
				saver.save(sess, "./ckpt/my_model")

			if global_step_value%FLAGS.eval_freq==0:
				print(en_eval)
				prediction = sess.run(transformer.prediction, feed_dict={transformer.input_en: en_eval,\
																		transformer.input_de: de_eval})
				fc_output_en = sess.run(transformer.fc_output_en, feed_dict={transformer.input_en: en_eval,\
																		transformer.input_de: de_eval})
				fc_output_de = sess.run(transformer.fc_output_de, feed_dict={transformer.input_en: en_eval,\
																		transformer.input_de: de_eval})
				# print("fc_output_en:"+str(fc_output_en[0][0]))
				# print("fc_output_de:"+str(fc_output_de[0][0]))
				# print("prediction result:"+str(prediction[0]))
				# bleu_score = bleu(prediction, de_eval)
				translate_ori = [[en_indexer[char] for char in batch] for batch in en_eval]
				translate_obj = [[de_indexer[char] for char in batch] for batch in de_eval]
				translate_resl = [[de_indexer[char] for char in batch] for batch in prediction]
				print("The original sentence:" + str(translate_ori[0]))
				print("The objective sentence:" + str(translate_obj[0]))
				print("The result sentence:" + str(translate_resl[0]))
def main(args=None):
	train(args)
if __name__ == "__main__":
	tf.app.run()