'''
# Implementation of MobileNet V3
# in TensorFlow (Model Define file)
# Author: Will@Alt****e
# Date: 2019/06/02
'''
from __future__ import division, print_function
import tensorflow as tf
import numpy as np
from TensorFlow_Utils import MobileNetV3_small as mnv3s_model
from TensorFlow_Utils import MobileNetV3_large as mnv3l_model
import cv2
import random
import copy

def parse_line(line):
	s = line.strip().split(' ')
	pic_path = s[0]
	# print(pic_path)
	label = s[1]
	img = cv2.imread(pic_path)/255.
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
	test_list = data_list[:1000]
	train_list = data_list[1000:]
	return train_list, test_list

def consume_data(img_lst, batch_size):
	images = []
	labels = []
	for i in range(0, min(batch_size, len(img_lst))):
		img, label = parse_line(img_lst.pop())
		images.append(img)
		labels.append(label)
	return images, labels

def convert_to_one_hot(labels):
	labels_len = len(labels)
	labels_one_hot = np.zeros([labels_len, 2])
	for idx in range(0, labels_len):
		if labels[idx]=='0':
			labels_one_hot[idx][0] = 1
		elif labels[idx]=='1':
			labels_one_hot[idx][1] = 1
	return labels_one_hot 

def evaluate_acc(pred_list, anno_list):
	# pred_list: np.array, "prediction list"
	# anno_list: char list, "annotation list"
	anno_list = np.array(anno_list)
	anno_list = anno_list.astype(np.int64)
	tp_count = 0
	fp_count = 0
	fn_count = 0
	tn_count = 0
	for idx in range(0, len(anno_list)):
		if ((anno_list[idx]==1) and (pred_list[idx]==1)):
			tp_count = tp_count + 1
		elif ((anno_list[idx]==0) and (pred_list[idx]==1)):
			fp_count = fp_count + 1
		elif ((anno_list[idx]==1) and (pred_list[idx]==0)):
			fn_count = fn_count + 1
		elif ((anno_list[idx]==0) and (pred_list[idx]==0)):
			tn_count = tn_count + 1
	if tp_count==0 and fp_count==0 :
		info = "precision: 0.000, recall: 0.000"
		print(info)		
		return 
	elif tp_count==0 and fn_count==0 :
		info = "precision: 0.000, recall: 0.000"
		print(info)		
		return 		
	precision =  tp_count / (tp_count + fp_count)
	recall = tp_count / (tp_count + fn_count)
	info = "precision: {:.3f}, recall: {:.3f}".format(precision, recall)
	print(info)
	return

def train_multi_gpu():
	def average_gradients(tower_grads):
		average_grads = []
		for grad_and_vars in zip(*tower_grads):
			grads =[]
			for g, _ in grad_and_vars:
				expanded_g = tf.expand_dims(g, 0)
				grads.append(expanded_g, 0)

			grad = tf.concat(grads, 0)
			grad = tf.reduce_mean(grad, 0)

			v = grad_and_vars[0][1]
			grad_and_var = (grad, v)
			average_grads.append(grad_and_var)
		return average_grads

	def average_losses(tower_losses):
		return tf.reduce_mean(tower_losses, axis=0)

	def assign_to_device(device, ps_device='/cpu:0'):
		def _assign(op):
			node_def = op if isinstance(op, tf.NodeDef) else op.node_def
			if node_def.op in PS_OPS:
				return ('/' + ps_device)
			else:
				return device
		return _assign

	image_list = parse_data('cuhkpq_train.txt')
	train_list, test_list = divide_data(image_list)
	train_list_backup = copy.deepcopy(train_list)
	test_list_backup = copy.deepcopy(test_list)
	batch_size = 128
	train_len = len(train_list)
	use_pretrained = 0
	num_gpus = 8

	save_dir = './checkpoints/'

	with tf.device('/cpu:0'):
		tower_grads = []
		tower_losses = []

		input_images = tf.placeholder(shape=[None, 224, 224, 3], dtype=tf.float32)
		input_labels = tf.placeholder(shape=[None, 2], dtype=tf.float32)

		for gpu_idx in range(num_gpus):
			with tf.device(assign_to_device('/gpu:{}'.format(i), ps_device='/cpu:0')):
				_input_images = input_images[gpu_idx * batch_size: (gpu_idx+1) * batch_size]
				_input_labels = input_labels[gpu_idx * batch_size: (gpu_idx+1) * batch_size]

				pred = mnv3s_model(_input_images)

				loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=_input_labels, logits=pred))

				optimizer = tf.train.RMSPropOptimizer(0.0005)
				
				if (gpu_idx==0):
					pred_eval = mnv3s_model(_input_images)
					saver_to_save = tf.train.Saver()
				saver_to_restore = tf.train.Saver()

				grads = optimizer.compute_gradients(loss)
				tower_grads.append(grads)
				tower_losses.append(loss)

		tower_grad = average_gradients(tower_grads)
		tower_loss = average_losses(tower_losses)
		train_op = optimizer.apply_gradients(tower_grads)

		init = tf.global_variables_initializer()

		epoches = 3000
		global_step = 1

		with tf.Session() as sess:
			sess.run(init)
			if use_pretrained:
				saver_to_restore.restore(sess, './checkpoints/mnv3.ckpt')
			for i in range(1， epoches+1):
				train_list = copy.deepcopy(train_list_backup)
				for j in range(0, train_len//(batch_size*num_gpus)):
					images, labels = consume_data(train_list, batch_size*num_gpus)
					labels = convert_to_one_hot(labels)
					_, loss_ = sess.run([train_op, tower_loss], feed_dict={input_images:images, input_labels:labels})

					info = "Epoch: {}, global_step: {}, average_loss: {:.3f}".format(i, global_step, loss_)
					print(info)

					if (global_step%200 == 0):
						pred_lst = []
						anno_lst = []
						test_list = copy.deepcopy(test_list_backup)
						for m in range(0, 100):
							images, labels = consume_data(test_list, 10)
							predictions = sess.run(pred_eval, feed_dict={input_images:images})
							predictions = np.argmax(prediction, axis=1)
							pred_lst.extend(prediction)
							anno_lst.extend(labels)
						evaluate_acc(pred_lst, anno_lst)

					if (global_step%1000 == 0):
						saver_to_save.save(sess, save_dir+'model-step_{}_loss_{:4f}'.format(global_step, loss_))

					global_step = global_step + 1

def train_single_gpu():
	image_list = parse_data('cuhkpq_train.txt')
	train_list, test_list = divide_data(image_list)
	train_list_backup = copy.deepcopy(train_list)
	test_list_backup = copy.deepcopy(test_list)
	batch_size = 32
	train_len = len(train_list)
	use_pretrained = 0

	save_dir = './checkpoints/'

	# Single GPU Version
	input_images = tf.placeholder(shape=[None, 224, 224, 3], dtype=tf.float32)
	input_labels = tf.placeholder(shape=[None, 2], dtype=tf.float32)

	pred = mnv3s_model(input_images)

	loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=input_labels, logits=pred))

	update_vars = tf.contrib.framework.get_variables_to_restore()

	optimizer = tf.train.RMSPropOptimizer(0.001)
	grads = optimizer.compute_gradients(loss, var_list=update_vars)
	train_op = optimizer.apply_gradients(grads)

	saver_to_save = tf.train.Saver()
	saver_to_restore = tf.train.Saver(var_list=update_vars)

	init = tf.global_variables_initializer()

	epoches = 3000
	global_step = 1

	with tf.Session() as sess:
		sess.run(init)
		if use_pretrained:
			saver_to_restore.restore(sess, './checkpoints/mnv3.ckpt')
		for i in range(1, epoches+1):
			train_list = copy.deepcopy(train_list_backup)
			random.shuffle(train_list)
			for j in range(0, int(train_len/(batch_size*num_gpus))):
				images, labels = consume_data(train_list, batch_size*num_gpus)
				labels = convert_to_one_hot(labels)
				_, pred_, loss_ = sess.run([train_op, pred, loss], feed_dict={input_images:images, input_labels:labels})
				
				info = "Epoch: {}, global_step: {}, loss: {:.3f}".format(i, global_step, loss_)
				print(info)

				if (global_step%200 == 0):
					pred_lst = []
					anno_lst = []
					test_list = copy.deepcopy(test_list_backup)
					for m in range(0, 100):
						images, labels = consume_data(test_list, 10)
						predictions = sess.run([pred], feed_dict={input_images:images})
						predictions = np.argmax(predictions[0], axis=1)
						pred_lst.extend(predictions)
						anno_lst.extend(labels)
					evaluate_acc(pred_lst, anno_lst)

				if (global_step%1000 == 0):
					saver_to_save.save(sess, save_dir+'model-step_{}_loss_{:4f}'.format(global_step, loss_))

				global_step = global_step + 1

def eval_pretrained():
	# parse .txt file
	image_list = parse_data('cuhkpq_test.txt')
	random.shuffle(image_list)
	test_list = image_list
	ckpt_dir = './checkpoints/'
	batch_size = 10

	# Single GPU Version
	input_images = tf.placeholder(shape=[None, 224, 224, 3], dtype=tf.float32)
	input_labels = tf.placeholder(shape=[None], dtype=tf.int32)

	pred = mnv3s_model(input_images)
	
	saver_to_restore = tf.train.Saver()

	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		saver_to_restore.restore(sess, './checkpoints/mnv3.ckpt')

		pred_lst = []
		anno_lst = []
		#test_list = image_list
		for m in range(0, len(test_list)//batch_size):
			images, labels = consume_data(test_list, batch_size)
			predictions = sess.run([pred], feed_dict={input_images:images})
			predictions = np.argmax(predictions[0], axis=1)
			pred_lst.extend(predictions)
			anno_lst.extend(labels)
		evaluate_acc(pred_lst, anno_lst)
				
if __name__ == '__main__':
	train_single_gpu()
	# train_multi_gpu()
	# eval_pretrained()
