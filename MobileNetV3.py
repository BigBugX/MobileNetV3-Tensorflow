'''
# Implementation of MobileNet V3
# in Tensorflow (Model Define file)
# Author: Will@Alt****e
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
	# print(pic_path)
	label = s[1]
	img = cv2.imread(pic_path)
	org_shape = img.shape
	org_h, org_w = org_shape[0], org_shape[1]
	if len(org_shape)<3 :
		tmp = np.zeros([org_w, org_h, 3])
		tmp[:,:,0] = img
		tmp[:,:,1] = img
		tmp[:,:,2] = img
		img = tmp
	img = cv2.resize(img, (224,224))
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

def evaluate(pred_list, anno_list):
	# pred_list: np.array
	# anno_list: char list
	anno_list = np.array(anno_list)
	anno_list = anno_list.astype(np.int64)
	difference = anno_list - pred_list
	difference_abs = np.absolute(difference)
	difference_sum = np.sum(difference_abs)
	return difference_sum/5000.

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
		for i in range(1, epoches+1):
			for j in range(0, int(train_len/batch_size)):
				images, labels = consume_data(train_list, batch_size)
				_, pred_, loss_ = sess.run([train_op, pred, loss], feed_dict={input_images:images, input_labels:labels})
				
				info = "Epoch: {}, global_step: {}, loss: {:.3f}".format(i, global_step, loss_)
				print(info)

				if (global_step%200 == 0):
					pred_lst = []
					anno_lst = []
					for m in range(0, 500):
						images, labels = consume_data(test_list, 10)
						predictions = sess.run([pred], feed_dict={input_images:images})
						predictions = np.argmax(predictions[0], axis=1)
						pred_lst.extend(predictions)
						anno_lst.extend(labels)
					accuracy = evaluate(pred_lst, anno_lst)
					info = "global_step: {}, accuracy: {:.3f}".format(global_step, accuracy)
					print(info)

				global_step = global_step + 1


