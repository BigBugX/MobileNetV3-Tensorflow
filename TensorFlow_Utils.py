'''
# Implementation of MobileNet V3
# in Tensorflow (Utils file)
# Author: Will@Alt****e
# Date: 2019/06/02
'''

import tensorflow as tf
import cv2

def hswish(input, 
	   name=None
	   ):
	output = input * tf.nn.relu6(features=(input+3)/6, name=name) 
	return output

def hsigmoid(input, 
	     name=None
	    ):
	output = tf.nn.relu6(features=(input+3)/6, name=name)
	return output


def GlobalAvgPool2D(input,
		    name=None
		    ):
	with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
		output = tf.keras.layers.GlobalAveragePooling2D(name='KerasGlobalAvgPooling2D')(input)
	return output

def Conv2D(input,
	   in_channels,
	   out_channels,
	   kernel_size=1,
	   stride=1,
	   padding='VALID',
	   name=None
   	   ):
	with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
		w_shape = [kernel_size, kernel_size, in_channels, out_channels]
		strides = [1, stride, stride, 1]
		filter =  tf.get_variable(name='weights', shape=w_shape, initializer=tf.initializers.glorot_normal, trainable=True)
		output = tf.nn.conv2d(input, filter, strides, padding='VALID', name='conv2d')
		return output

def BatchNorm(input,
	      name=None
	      ):
	with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
		input_shape = input.get_shape().as_list()
		params_shape = input_shape[-1]
		axis = list(range(len(input_shape)-1))

		beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer())
		gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer())
		# moving_mean = tf.get_variable('moving_mean', params_shape, initializer=tf.zeros_initializer())
		# moving_variance = tf.get_variable('moving_variance', params_shape, initializer=tf.ones_initializer())

		mean, var = tf.nn.moments(input, axis)
		output = tf.nn.batch_normalization(input, mean, var, beta, gamma, variance_epsilon=0.001, name='batch_normalization')
	return output

def ReLU(input,
	 name=None
	 ):
	output = tf.nn.relu(input,name=name)
	return output

def SeBlock(input,
   	    in_channels,
	    reduction,
    	    name=None	
	    ):
	with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
		out_channels = in_channels // reduction
		# Spatial-Excitation Module
		# output = GlobalAvgPool2D(input, name='GlobalAvgPool2D')
		output = Conv2D(input, in_channels, out_channels, name='Conv2D_Se1')
		output = BatchNorm(output, name='BatchNorm_Se1')
		output = ReLU(output, name='ReLU')
		output = Conv2D(output, out_channels, in_channels, name='Conv2D_Se2')
		output = BatchNorm(output, name='BatchNorm_Se2')
		output = hsigmoid(output, name='hsigmoid')
		output = input * output
	return output

def ResBlock(input,
	     in_channels,
   	     out_channels,
   	     kernel_size=1,
	     stride=1,
   	     padding='VALID',
	     name=None
	     ):
	with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
		output = Conv2D(input, in_channels, out_channels, 1, 1, 0, name='Conv2D')
		output = BatchNorm(input, name='BatchNorm')
	return output

def Block(input,
	  in_channels,
	  out_channels,
	  ex_channels,
	  kernel_size,
	  nonlinear,
	  seblock,
	  stride,
	  name=None
	  ):
	with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
		# strides = [1, stride, stride, 1]
		# conv1
		output = Conv2D(input, in_channels, ex_channels, 1, 1, 0, name=name+'Conv2D_1')
		output = BatchNorm(output, name='BatchNorm_1')
		if nonlinear=='ReLU':
			output = ReLU(output, name='ReLU_1')
		elif nonlinear=='hswish':
			output = hswish(output, name='hswish_1')
		# conv2
		output = Conv2D(output, ex_channels, ex_channels, kernel_size, stride, padding='VALID', name=name+'Conv2D_2')
		output = BatchNorm(output, name='BatchNorm_2')
		if nonlinear=='ReLU':
			output = ReLU(output, name='ReLU_2')
		elif nonlinear=='hswish':
			output = hswish(output, name='hswish_2')
		# conv3
		output = Conv2D(output, ex_channels, out_channels, 1, 1, 0, name=name+'Conv2D_3')
		output = BatchNorm(output, name='BatchNorm_3')
		# shortcut
		if seblock:
			output = SeBlock(output, in_channels=in_channels, reduction=4, name='SeBlock')
		if stride == 1:
			output = output + ResBlock(output, in_channels, out_channels, name='ResBlock')
	return output

def MobileNetV3_small(input,
		      in_channels=3,
	  	      out_channels=16,
		      kernel_size=3,
		      stride=2,
		      padding='SAME',
		      name='MobileNetV3_small'
		      ):
	with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
		# conv1
		output = Conv2D(input, in_channels, out_channels, 3, 2, 'SAME', name='Conv2D_1')
		output = BatchNorm(output, name='BatchNorm_1')
		output = hswish(output, name='hswish_1')
		# bottle_neck output
		#                          in  ou  ex  k    nl    se  s
		b_output = Block(output,   16, 16, 16, 3, 'ReLU', 16, 2,    name='block1')
		b_output = Block(b_output, 16, 24, 72, 3, 'ReLU', None, 2,  name='block2')
		b_output = Block(b_output, 24, 24, 88, 3, 'ReLU', None, 1,  name='block3')
		b_output = Block(b_output, 24, 40, 96, 5, 'hswish', 40, 2,  name='block4')
		b_output = Block(b_output, 40, 40, 240, 5, 'hswish', 40, 1, name='block5')
		b_output = Block(b_output, 40, 40, 240, 5, 'hswish', 40, 1, name='block6')
		b_output = Block(b_output, 40, 48, 120, 5, 'hswish', 48, 1, name='block7')
		b_output = Block(b_output, 48, 48, 144, 5, 'hswish', 48, 1, name='block8')
		b_output = Block(b_output, 48, 96, 288, 5, 'hswish', 96, 2, name='block9')
		b_output = Block(b_output, 96, 96, 576, 5, 'hswish', 96, 1, name='block10')
		b_output = Block(b_output, 96, 96, 576, 5, 'hswish', 96, 1, name='block11')
		# conv2
		output = Conv2D(b_output, 96, 576, 1, 1, 'VALID', name='Conv2D_2')
		output = BatchNorm(output, name='BatchNorm_2')
		output = hswish(output, name='hswish_2')
		# avg_pool
		output = tf.nn.avg_pool(output, ksize=[1,7,7,1], strides=[1,1,1,1], padding='VALID')

		output = Conv2D(output, 576, 1280, 1, 1, 'VALID', name='Conv2D_3')
		output = BatchNorm(output, name='BatchNorm_3')
		output = hswish(output, name='hswish_3')
		output = tf.keras.layers.Flatten()(output)
		output = tf.keras.layers.Dense(2, activation='relu')(output)

	return output






