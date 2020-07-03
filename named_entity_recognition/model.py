import tensorflow as tf
class BiLSTM_CRF():
	def __init__(self, batch_size, embedding_size, hidden_size,
	 				lr, vocab_size, tag_num):

		with tf.name_scope("embedding_layer"):
			self.input_sents = tf.placeholder(dtype=tf.int64, shape=(batch_size, None), name="input_sents")
			self.input_labels = tf.placeholder(dtype=tf.int64, shape=(batch_size, None), name="input_labels")
			self.sequence_lengths = tf.placeholder(dtype=tf.int64, shape=(batch_size), name="sequence_lengths")
			reshape_labels = tf.reshape(self.input_labels, [-1])
			embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="embedding")
			embed_sents = tf.nn.embedding_lookup(embedding, self.input_sents)
			self.global_step = tf.Variable(0, trainable=False, name="global_step")
		
		with tf.name_scope("bilstm_layer"):
			fw = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
			bw = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
			(output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
				cell_fw = fw, 
				cell_bw = bw, 
				inputs = embed_sents, 
				sequence_length = self.sequence_lengths,
				dtype = tf.float32)
			lstm_output = tf.concat([output_fw, output_bw], 2)
			#[batch_size, max_time, hidden_size*2]

		with tf.name_scope("fc_layer"):
			W = tf.get_variable(name="W",
							shape=[hidden_size*2, tag_num],
							initializer=tf.contrib.layers.xavier_initializer())
			b = tf.get_variable(name="b",
							shape=[tag_num],
							initializer=tf.zeros_initializer())
			lstm_output = tf.reshape(lstm_output, [-1, hidden_size*2])
			logits = tf.matmul(lstm_output, W) + b
			reshape_logits = tf.reshape(logits, [batch_size, -1, tag_num])

		with tf.name_scope("loss"):
			self.likelihood, _ = tf.contrib.crf.crf_log_likelihood(inputs=reshape_logits,
												tag_indices=self.input_labels,
												sequence_lengths=self.sequence_lengths)
			self.loss = tf.negative(tf.reduce_mean(self.likelihood), name="loss")
			self.train_step = tf.train.AdamOptimizer(lr).minimize(self.loss, global_step=self.global_step, name="train_step")

		with tf.name_scope("accuracy"):
			self.result = tf.argmax(logits, axis=1, name="result")
			self.shape = tf.shape(self.result, name="shape")
			equal = tf.equal(self.result, reshape_labels)
			result = tf.cast(self.result, bool)
			self.right_ner = tf.reduce_sum(tf.cast(tf.logical_and(equal, result), tf.int64), name="right_ner")
			self.recog_ner = tf.reduce_sum(tf.cast(tf.not_equal(self.result, 0), tf.int64), name="recog_ner")
			self.all_ner = tf.reduce_sum(tf.cast(tf.not_equal(reshape_labels, 0), tf.int64), name="all_ner")
			p = self.right_ner/self.recog_ner
			r = self.right_ner/self.all_ner
			self.f_score = tf.multiply(tf.cast(2, tf.float64), p*r/(p+r), name="f_score")