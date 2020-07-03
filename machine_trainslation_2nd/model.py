import tensorflow as tf
import numpy as np
import math
class Transformer:
	def __init__(self, vocab_size_en, vocab_size_de, max_len_en, max_len_de,
	 embedding_size, h, c_size, en_num, de_num, lr):
		def embedding(input_batch, embedding_table, max_len, embedding_size):
			#input_batch: batch_size, max_len, embedding_size
			embedding_ids = tf.nn.embedding_lookup(embedding_table, input_batch)
			masking = tf.sign(tf.abs(tf.reduce_sum(embedding_ids, axis=-1)))
			batch_size = tf.expand_dims(tf.shape(input_batch)[0], axis=0)
			tile_shape = tf.concat([batch_size, tf.convert_to_tensor([1,1])], axis=0)
			embedding_pos = []
			for pos in range(max_len):
				embedding_pos_list = []
				# print("embedding_size: "+str(embedding_size))
				# print("max_len: "+str(max_len))
				for i in range(embedding_size):
					if i%2==0:
						embedding_pos_list.append(math.sin(pos/math.pow(float(10000), i/embedding_size)))
					else:
						embedding_pos_list.append(math.cos(pos/math.pow(float(10000), i/embedding_size)))
				embedding_pos.append(embedding_pos_list)
			# print(np.shape(np.array(embedding_pos)))
			# print("I love you, too.")
			# exit()
			#input_batch: batch_size, max_len, embedding_size
			embedding_pos_tensor = tf.convert_to_tensor(embedding_pos, dtype=tf.float32)
			embedding_pos_tensor = tf.tile(tf.expand_dims(embedding_pos_tensor, axis=0), tile_shape)
			output = tf.add(embedding_ids, embedding_pos_tensor)
			output *= tf.tile(tf.expand_dims(masking, -1), [1, 1, embedding_size])
			#print("Embedding part of graph finished.")
			return (output, masking)

		def multi_head_attention(query, key, value, h, max_len, query_masking, key_masking, future_blinding = False):
			# query shape: batch_size, d_query, c_size
			# key shape: batch_size, d_key, c_size
			# value shape: batch_size, d_key, c_size
			# suppose d_query=d_key=d_input
			
			c_size = query.get_shape()[2]
			d_query = query.get_shape()[1]
			d_key = key.get_shape()[1]
			query = tf.layers.dense(query, c_size*h, use_bias=True)
			key = tf.layers.dense(key, c_size*h, use_bias=True)
			value = tf.layers.dense(value, c_size*h, use_bias=True)
			multi_query = multi_head_split(query, h)
			multi_key = multi_head_split(key, h)
			multi_value = multi_head_split(value, h)

			#multi shape: h, batch_size, d_input, c_size
			transposed_key = tf.transpose(multi_key, [0, 1, 3, 2])
			#h, batch_size, c_size, d_input
			# print("transposed_key:"+str(transposed_key.get_shape()))
			dot_attention = tf.linalg.matmul(multi_query, transposed_key)
			# print("dot_attention:"+str(dot_attention))
			#h, batch_size, d_query, c_size * h, batch_size, c_size, d_key
			#=h, batch_size, d_query, d_key

			# [h, batch_size, d_input]
			# print("query_masking:"+str(query_masking.get_shape()))
			key_masking = tf.tile(tf.expand_dims(key_masking, 0), [h, 1, 1])
			key_masking = tf.tile(tf.expand_dims(key_masking, -1), [1, 1, 1, d_query])
			key_masking = tf.transpose(key_masking, [0, 1, 3, 2])
			#[h, batch_size, d_query, d_key]

			dot_conts = tf.ones_like(dot_attention)*(-2**31+1)

			dot_attention = tf.where(tf.equal(key_masking, 0), dot_attention, dot_conts)
			# print("dot_attention:"+str(dot_attention.get_shape()))

			# dot_attention = dot_attention * query_masking
			# dot_attention = dot_attention + query_adding
			# #print("Query masking part of graph finished.")

			dot_attention = tf.math.divide(dot_attention, tf.math.pow(tf.to_float(d_key), 0.5))
			if future_blinding == True:
				future_masking = []
				for i in range(d_query):
					future_masking_i = [1 for _ in range(i)] + [0 for _ in range(d_key-i)]
					future_masking.append(future_masking_i)
				future_masking = tf.convert_to_tensor(future_masking, dtype=tf.float32)
				future_masking = future_masking * key_masking
				# [h, batch_size, d_query, d_key] * [d_query, d_key]
				# print("dot_attention:"+str(dot_attention.get_shape()))
				# print("future_masking"+str(future_masking.get_shape()))
				dot_attention = tf.where(tf.equal(future_masking, 0), dot_attention, dot_conts)
				# print("dot_attention after future blinding:"+str(dot_attention.get_shape()))
			#print("Future blinding part of graph finished.")

			dot_attention = tf.nn.softmax(dot_attention)

			query_masking = tf.tile(tf.expand_dims(query_masking, -1), [1, 1, d_key])
			query_masking = tf.tile(tf.expand_dims(query_masking, 0), [h, 1, 1, 1])
			dot_attention = tf.multiply(dot_attention, query_masking)
			# [h, batch_size, d_query, d_key] * [h, batch_size, d_query, d_key]
			# print("dot_attention after key masking:"+str(dot_attention.get_shape()))

			multi_attention = tf.linalg.matmul(dot_attention, multi_value)
			#[h, batch_size, d_query, d_key] * [h, batch_size, d_key, c_size]
			#=[h, batch_size, d_query, c_size]
			multi_attention = tf.concat(tf.split(multi_attention, h, axis=0), axis=-1)
			#[1, batch_size, d_query, c_size*h]
			multi_attention = tf.squeeze(multi_attention, [0])
			#[batch_size, d_query, c_size*h]
			multi_attention = tf.layers.dense(multi_attention, int(c_size), use_bias=True)
			#print("Multi head attention part of graph finished.")
			return multi_attention

		def multi_head_split(input_tensor, h):
			#input tensor: batch_size, d_input, c_size
			reshape_tensor = tf.concat(tf.split(tf.expand_dims(input_tensor, 0), h, axis=-1), 0)
			return reshape_tensor
			#h, batch_size, d_input, c_size/h

		def layer_normalization(input_batch, epsilon=1e-5):
			mean, variance = tf.nn.moments(input_batch, axes=[-1], keep_dims=True)
			output_batch = (input_batch-mean)/tf.math.sqrt(variance+epsilon)
			return output_batch

		def feed_forward(input_batch, conv_sizes=[2048, 512]):
			# input_batch: batch_size, d_input, c_size
			mid_layer = tf.layers.conv1d(input_batch, filters=conv_sizes[0], kernel_size=1,\
			 activation=tf.nn.relu, use_bias=True)
			output_layer = tf.layers.conv1d(mid_layer, filters=conv_sizes[1], kernel_size=1,\
			 activation=None, use_bias=True)
			#print("Feed forward part of graph finished.")
			return output_layer

		with tf.name_scope("embedding_part"):
			self.embedding_table_en = tf.get_variable(shape=[vocab_size_en+3, embedding_size], \
				initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32), dtype=tf.float32, name="embedding_table_en")
			self.embedding_table_de = tf.get_variable(shape=[vocab_size_de+3, embedding_size], \
				initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32), dtype=tf.float32, name="embedding_table_de")
			self.input_en = tf.placeholder(dtype=tf.int32, shape=[None, max_len_en], name="input_en")
			self.input_de = tf.placeholder(dtype=tf.int32, shape=[None, max_len_de], name="input_de")
			# notice the last batch may do not have full size
			# so we do not use the batch_size in tf.FLAGS

			embedded_en, masking_en = embedding(self.input_en, self.embedding_table_en, max_len_en, embedding_size)
			embedded_de, masking_de = embedding(self.input_de, self.embedding_table_de, max_len_de, embedding_size)
			densed_en = tf.layers.dense(embedded_en, c_size, use_bias=True)
			densed_de = tf.layers.dense(embedded_de, c_size, use_bias=True)
			self.densed_en = densed_en * tf.tile(tf.expand_dims(masking_en, -1), [1,1,c_size])
			self.densed_de = densed_de * tf.tile(tf.expand_dims(masking_de, -1), [1,1,c_size])
			print("embedding part finished")

		fc_output_en = self.densed_en
		for i in range(en_num):
			with tf.name_scope("encoder_"+str(i)):
				att_input_en = fc_output_en
				query = key = value = att_input_en
				att_mid_en = multi_head_attention(query, key, value, h, max_len_en, masking_en, masking_en)
				att_output_en = att_input_en + att_mid_en
				att_output_en = layer_normalization(att_output_en, 1e-5)
				fc_input_en = att_output_en
				fc_mid_en = feed_forward(fc_input_en)
				fc_output_en = fc_input_en + fc_mid_en
				fc_output_en = layer_normalization(fc_output_en, 1e-5)
				print("one encoder finished")
		self.fc_output_en = fc_output_en

		fc_output_de = self.densed_de
		for i in range(de_num):
			with tf.name_scope("decoder_"+str(i)):
				self_att_input_de = fc_output_de
				query = key = value = self_att_input_de
				self_att_mid_de = multi_head_attention(query, key, value, h, max_len_de, masking_de, masking_de, future_blinding=True)
				self_att_output_de = self_att_input_de + self_att_mid_de
				self_att_output_de = layer_normalization(self_att_output_de)
				att_input_de = self_att_output_de
				query = att_input_de
				key = value = fc_output_en
				att_mid_de = multi_head_attention(query, key, value, h, max_len_de, masking_de, masking_en)
				att_output_de = att_input_de + att_mid_de
				att_output_de = layer_normalization(att_output_de)
				fc_input_de = att_output_de
				fc_mid_de = feed_forward(fc_input_de)
				fc_output_de = fc_input_de + fc_mid_de
				fc_output_de = layer_normalization(fc_output_de, 1e-5)
				print("one decoder finished")
		self.fc_output_de = fc_output_de

		with tf.name_scope("final_part"):
			final_output = tf.layers.dense(fc_output_de, vocab_size_de+4, activation=tf.nn.relu)
			#batch_size, max_len, vocab_size_de+2
			self.prediction = tf.argmax(final_output, dimension=-1, output_type=tf.int32)
			one_hot_label = tf.one_hot(self.input_de, depth=vocab_size_de+4)
			#print(one_hot_label)
			self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_label, logits=final_output)
			self.loss = tf.math.reduce_sum(self.loss, name="loss")
			self.global_step = tf.Variable(0, trainable=False, name='global_step')
			self.optimizer = tf.train.AdamOptimizer(learning_rate = lr, name="optimizer")
			self.train_step = self.optimizer.minimize(self.loss, global_step=self.global_step, name="train_step")