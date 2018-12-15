import tensorflow as tf
import numpy as np
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


class TextCNN:

    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # sequence_length     최대 문자열 길이
        # num_classes         긍정 / 부정 (2)
        # vocab_size          총 단어 개수
        # embedding_size      문자 임베딩 최대 길이(128)
        # filter_sizes        (3, 4, 5)
        # num_filters         크기 당 필터 개수(128)
        # l2_reg_lambda       l2 정규화 (건드릴 일 없음)

        self.input_x = tf.placeholder(
            tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(
            tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name="dropout_keep_prob")

        with tf.device('/gpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(tf.random_uniform(
                [vocab_size, embedding_size], -1.0, 1.0), name='W')
            self.embedded_chars = tf.nn.embedding_lookup
            self.embedded_chars_expanded = tf.expand_dims(
                self.embedded_chars, -1)

        pooled_outputs = []

        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(
                    filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(
                    0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID"
                )

                h = tf.nn.relu(tf.nn.bias_add(conv, b))
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID'
                )
                pooled_outputs.append(pooled)
