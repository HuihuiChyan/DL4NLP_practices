import tensorflow as tf
import pdb
class NN_MODEL():
	def __init__(self, config):
		self.config = config
		self._init_parameters()
		self._init_placeholder()
		self._build_graph()

	def _build_graph(self):
		# pdb.set_trace()
		self.embedding = tf.get_variable(initializer = tf.random_uniform([self.vocab_size, self.embedding_size], -1., 1.),\
		 dtype=tf.float32, name='embedding')
		input_encode = tf.nn.embedding_lookup(self.embedding, self.input_sent)
		# [batch_size, max_len, embedding_size]
		middle = tf.layers.dense(input_encode, self.hidden_size,\
			activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
		# [batch_size, max_len, hidden_size]
		self.out = tf.layers.dense(middle, self.classes, kernel_initializer=tf.contrib.layers.xavier_initializer())
		# [batch_size, max_len, classes]
		predict = tf.argmax(self.out, axis=-1, output_type=tf.int32)

		self.accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, self.input_label), tf.float32))
		self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_label, logits=self.out)
		self.train = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

	def _init_parameters(self):
		self.maxlen = self.config['maxlen']
		self.lr = self.config['lr']
		self.vocab_size = self.config['vocab_size']
		self.embedding_size = self.config['embedding_size']
		self.dropout = self.config['dropout']
		self.classes = self.config['classes']
		self.hidden_size = self.config['hidden_size']

	def _init_placeholder(self):
		self.input_sent = tf.placeholder(dtype=tf.int32, shape=[None, self.maxlen])
		self.input_label = tf.placeholder(dtype=tf.int32, shape=[None, self.maxlen])