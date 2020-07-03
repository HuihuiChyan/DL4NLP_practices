import tensorflow as tf

class TextCNN:

	def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_size, num_filters, l2_reg_lambda, vocab, max_len):

		# _ = tf.Variable(initial_value = "Fake_Variable")
		self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, sequence_length], name="input_x")
		self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes], name="input_y")
		self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, name="dropout_keep_prob")
		self.vocab = tf.constant(vocab, name="vocab")
		self.max_len = tf.constant(sequence_length, name="max_len")

		with tf.name_scope("embedding"):
			W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
			embeded_chars = tf.nn.embedding_lookup(W, self.input_x)
			embeded_chars_expanded = tf.reshape(embeded_chars, [-1, sequence_length, embedding_size, 1])

		with tf.name_scope("conv_layer"):
			W = tf.Variable(tf.random_uniform(shape=[filter_size, embedding_size, 1, num_filters], dtype=tf.float32))
			b = tf.Variable(tf.random_uniform(shape=[num_filters], dtype=tf.float32))
			y = tf.nn.conv2d(embeded_chars_expanded, W, [1, 1, 1, 1], padding="VALID")
			#dimension of y: [batch_size, sequence_length-1, 1, num_filters]
			conv_output = tf.nn.relu(y+b)

		with tf.name_scope("max_pool_layer"):
			pool1_output = tf.nn.max_pool(conv_output,
				ksize = [1, sequence_length-filter_size+1, 1, 1],
				strides = [1, 1, 1, 1],
				padding = "VALID")
			#dimension of pool1_output:[batch_size, 1, 1, num_filters]
			pool1_reshape = tf.reshape(pool1_output, shape = [-1, num_filters])

		with tf.name_scope("dropout_layer"):
			dropout_output = tf.nn.dropout(pool1_reshape, keep_prob = self.dropout_keep_prob)

		with tf.name_scope("fc"):
			W = tf.Variable(tf.random_normal(shape=[num_filters, num_classes], dtype=tf.float32))
			b = tf.Variable(tf.random_normal(shape=[num_classes], dtype=tf.float32))
			scores = tf.matmul(dropout_output, W) + b
			#dimension of fc_output:[batch_size, num_classes]

		with tf.name_scope("accuracy"):
			self.predict = tf.argmax(scores, 1, name="predict")
			self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predict, tf.argmax(self.input_y, 1)), dtype=tf.float32), name="accuracy")
		
		with tf.name_scope("loss"):
			#self.loss = tf.reduce_mean(-tf.reduce_sum(self.input_y*tf.log(fc_output), reduction_indices=[1]))
			self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=scores))
