'''
# Implementation of MobileNet V3
# in Tensorflow (Model Define file)
# Author: Will@Altizure
# Date: 2019/06/02
'''
from __future__ import division, print_function
import tensorflow as tf
import numpy as np
from TensorFlow_Utils import MobileNetV3_small as mnv3s_model
import cv2
import random

def parse_line(line):
	s = line.strip().split(' ')
	pic_path = s[0]
	label = s[1]
	img = cv2.imread(pic_path)
	img = cv2.resize([224,224])
	return img, label

def parse_data(file_path):
	img_lst = []
	with open(file_path, 'r') as f:
		img_lst = f.readlines()
	return img_lst

def divide_data(data_list):
	random.shuffle(data_list)
	test_list = data_list[:5000]
	train_list = data_list[5000:]
	return train_list, test_list

def consume_data(img_lst, batch_size):
	images = []
	labels = []
	for i in range(0, min(batch_size, len(img_lst))):
		img, label = parse_line(img_lst.pop())
		images.append(img)
		labels.append(label)
	return images, labels

if __name__ == '__main__':

	image_list = parse_data('cuhkpq.txt')
	train_list, test_list = divide_data(image_list)
	batch_size = 16
	train_len = len(train_list)


	# Single GPU Version
	input_images = tf.placeholder(shape=[None, 224, 224, 3], dtype=tf.float32)
	input_labels = tf.placeholder(shape=[None, 1], dtype=tf.int32)

	pred = mnv3s_model(input_images)
	loss = tf.losses.sparse_softmax_cross_entropy(labels=input_labels, logit=pred)

	optimizer = tf.train.Adam(0.001)
	train_op = optimizer.minimize(loss)

	init = tf.global_variables_initializer()

	epoches = 100
	global_step = 1

	with tf.Session() as sess:
		sess.run(init)
		for i in range(0, epoches):
			for i in range(0, train_len/batch_size):
				images, labels = consume_data(train_list, batch_size)
				_, loss_ = sess.run([train_op, loss], feed_dict={input_images:images, input_labels:labels})
				info = "Epoch: {}, global_step: {}, loss: {:.3f}".format(i, global_step, loss_)
				global_step = global_step + 1


