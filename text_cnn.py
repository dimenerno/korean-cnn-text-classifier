import tensorflow as tf
import numpy as np
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


class TextCNN:

    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        '''
        sequence_length     최대 문자열 길이
        num_classes         긍정 / 부정 (2)
        vocab_size          총 단어 개수
        embedding_size      ?
        filter_sizes        ?
        num_filters         ?
        l2_reg_lambda       l2 정규화 (건드릴 일 없음)
        '''

        self.input_x = tf.placeholder(
            tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(
            tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name="dropout_keep_prob")

        with tf.device('/gpu:0'), tf.name_scope("embedding"):
            pass
