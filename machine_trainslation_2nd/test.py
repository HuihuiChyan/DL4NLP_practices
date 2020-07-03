import tensorflow as tf 
import pickle
import jieba
tf.flags.DEFINE_integer("batch_size", 128, "The size of each batch.")
tf.flags.DEFINE_integer("num_epoch", 100, "Number epoches.")
tf.flags.DEFINE_boolean("ja2ch", False, "Whether to translate Japanese into Chinese.")
tf.flags.DEFINE_integer('eval_freq', 10, 'Evaluate after certain steps.')
tf.flags.DEFINE_integer('save_freq', 10, 'Save after certain steps.')
tf.flags.DEFINE_integer('max_len_ja', 50, 'Max length of Japanese sentences.')
tf.flags.DEFINE_integer('max_len_ch', 50, 'Max length of Chinese sentences.')
FLAGS = tf.flags.FLAGS
def build_test_line(input_line, vocab):
	output_line = []
	for char in line:
		if char in vocab.keys():
			output_line.append(vocab[char])
		else:
			output_line.append(1)
	return output_line
with open("./data/ch_vocab.txt", "rb") as ch_vocab_f,\
	open("./data/ja_vocab.txt", "rb") as ja_vocab_f:
	ch_vocab = pickle.loads(ch_vocab_f.read())
	ja_vocab = pickle.loads(ja_vocab_f.read())
if FLAGS.ja2ch == True:
	ja_sent = mecab.cut(input("Please input your Japanese sentence:"))
	ja_sent = ['BOS']+my_sent
	ja_sent = my_sent+['EOS']
	en_line = build_test_line(ja_sent, ja_vocab)
	en_vocab = ja_vocab
	de_vocab = ch_vocab
	max_len_en = FLAGS.max_len_ch
		
elif FLAGS.ja2ch == False:
	ch_sent = jieba.cut(input("Please input your Chinese sentence:"))
	ch_sent = ['BOS']+ch_sent
	ch_sent = ch_sent+['EOS']
	en_line = build_test_line(ch_sent, ch_vocab)
	en_vocab = ch_vocab
	de_vocab = ja_vocab
	max_len_de = FLAGS.max_len_ja
en_indexer = {v:k for k,v in en_vocab.items()}
de_indexer = {v:k for k,v in de_vocab.items()}
with tf.Session() as sess:
	ckpt = tf.train.get_checkpoint_state('./ckpt') 
	saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta')
	saver.restore(sess, ckpt.model_checkpoint_path)
	graph = tf.get_default_graph()
	input_en = graph.get_tensor_by_name("transformer/embedding_part/input_en:0")
	input_de = graph.get_tensor_by_name("transformer/embedding_part/input_de:0")
	prediction = graph.get_tensor_by_name("transformer/final_part/prediction:0")
	de_line = [['PAD' for i in range(max_len_de)]]
	en_line = [en_line]
	for i in range(max_len_de):
		prediction_result= sess.run(prediction,\
		 feed_dict={input_en: en_line, input_de: de_line})
		translate_resl = de_indexer[char] for char in de_line[0]
		if 'EOS' in translate_resl:
			print("The result sentence:" + str(translate_resl[0]))
			break
		de_line = prediction_result