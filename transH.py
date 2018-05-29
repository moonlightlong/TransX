#!/usr/bin/python
# -*- coding:utf-8 -*-
from __future__ import division, print_function, unicode_literals

import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers as tflayers

from model import Model


class TransH(Model):
    
    def __init__(self, kg, embedding_dim, margin_value, score_func,
                 batch_size, learning_rate, n_generator, n_rank_calculator):
        super(TransH, self).__init__(kg, embedding_dim, margin_value, score_func,
                                     batch_size, learning_rate, n_generator,
                                     n_rank_calculator)
    
    def build_graph(self):
        with tf.variable_scope('embedding'):
            self.entity_embedding = tf.get_variable(name='entity',
                                                    shape=[self.kg.n_entity, self.embedding_dim],
                                                    initializer=tflayers.xavier_initializer(uniform = False))
            tf.summary.histogram(name=self.entity_embedding.op.name, values=self.entity_embedding)
            self.relation_embedding = tf.get_variable(name='relation',
                                                      shape=[self.kg.n_relation, self.embedding_dim],
                                                      initializer=tflayers.xavier_initializer(uniform = False))
            tf.summary.histogram(name=self.relation_embedding.op.name, values=self.relation_embedding)
            self.normal_embedding = tf.get_variable(name='normal_vector',
                                                      shape=[self.kg.n_relation, self.embedding_dim],
                                                      initializer=tflayers.xavier_initializer(uniform = False))
            tf.summary.histogram(name=self.normal_embedding.op.name, values=self.normal_embedding)
        with tf.name_scope('normalization'):
            self.entity_embedding = tf.nn.l2_normalize(self.entity_embedding, axis=1)
            self.relation_embedding = tf.nn.l2_normalize(self.relation_embedding, axis=1)
            self.normal_embedding = tf.nn.l2_normalize(self.normal_embedding, axis=1)
        with tf.name_scope('training'):
            distance_pos, distance_neg = self.infer(self.triple_pos, self.triple_neg)
            self.loss = self.calculate_loss(distance_pos, distance_neg, self.margin)
            tf.summary.scalar(name=self.loss.op.name, tensor=self.loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
            self.merge = tf.summary.merge_all()
    
    def build_eval_graph(self):
        with tf.name_scope('evaluation'):
            self.idx_head_prediction, self.idx_tail_prediction = self.evaluate(self.eval_triple)
    
    def __transfer(self, e, normal):
        return e - tf.reduce_sum(e * normal, 1, keepdims=True) * normal
    
    def __transfer_(self, e, normal):
        return e - tf.reduce_sum(e * normal) * normal

    def infer(self, triple_pos, triple_neg):
        with tf.name_scope('lookup'):
            head_pos = tf.nn.embedding_lookup(self.entity_embedding, triple_pos[:, 0])
            tail_pos = tf.nn.embedding_lookup(self.entity_embedding, triple_pos[:, 1])
            relation_pos = tf.nn.embedding_lookup(self.relation_embedding, triple_pos[:, 2])
            normal_pos = tf.nn.embedding_lookup(self.normal_embedding, triple_pos[:, 2])

            head_neg = tf.nn.embedding_lookup(self.entity_embedding, triple_neg[:, 0])
            tail_neg = tf.nn.embedding_lookup(self.entity_embedding, triple_neg[:, 1])
            relation_neg = tf.nn.embedding_lookup(self.relation_embedding, triple_neg[:, 2])
            normal_neg = tf.nn.embedding_lookup(self.normal_embedding, triple_neg[:, 2])
        with tf.name_scope('link'):
            distance_pos = self.__transfer(head_pos, normal_pos) + relation_pos - self.__transfer(tail_pos, normal_pos)
            distance_neg = self.__transfer(head_neg, normal_neg) + relation_neg - self.__transfer(tail_neg, normal_neg)
        return distance_pos, distance_neg

    def evaluate(self, eval_triple):
        with tf.name_scope('lookup'):
            head = tf.nn.embedding_lookup(self.entity_embedding, eval_triple[0])
            tail = tf.nn.embedding_lookup(self.entity_embedding, eval_triple[1])
            relation = tf.nn.embedding_lookup(self.relation_embedding, eval_triple[2])
            normal_vec = tf.nn.embedding_lookup(self.normal_embedding, eval_triple[2])
        with tf.name_scope('link'):
            embedding = self.entity_embedding
            distance_head_prediction = self.__transfer(embedding, normal_vec) + relation - self.__transfer_(tail, normal_vec)
            distance_tail_prediction = self.__transfer_(head, normal_vec) + relation - self.__transfer(embedding, normal_vec)
        with tf.name_scope('rank'):
            if self.score_func == 'L1':  # L1 score
                _, idx_head_prediction = tf.nn.top_k(tf.reduce_sum(tf.abs(distance_head_prediction), axis=1),
                                                     k=self.kg.n_entity)
                _, idx_tail_prediction = tf.nn.top_k(tf.reduce_sum(tf.abs(distance_tail_prediction), axis=1),
                                                     k=self.kg.n_entity)
            else:  # L2 score
                _, idx_head_prediction = tf.nn.top_k(tf.reduce_sum(tf.square(distance_head_prediction), axis=1),
                                                     k=self.kg.n_entity)
                _, idx_tail_prediction = tf.nn.top_k(tf.reduce_sum(tf.square(distance_tail_prediction), axis=1),
                                                     k=self.kg.n_entity)
        return idx_head_prediction, idx_tail_prediction
