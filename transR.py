#!/usr/bin/python
# -*- coding:utf-8 -*-
from __future__ import division, print_function, unicode_literals

import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers as tflayers
from model import Model


class TransR(Model):
    
    def __init__(self, kg, embedding_dim, margin_value, score_func,
                 batch_size, learning_rate, n_generator, n_rank_calculator):
        if isinstance(embedding_dim, int):
            self.embedding_ent_dim = embedding_dim
            self.embedding_rel_dim = embedding_dim
        else:
            self.embedding_ent_dim = embedding_dim[0]
            self.embedding_rel_dim = embedding_dim[1]
        super(TransR, self).__init__(kg, embedding_dim, margin_value, score_func,
                                     batch_size, learning_rate, n_generator,
                                     n_rank_calculator)
    
    def build_graph(self):
        with tf.variable_scope('embedding'):
            self.entity_embedding = tf.get_variable(name='entity',
                                                    shape=[self.kg.n_entity, self.embedding_ent_dim],
                                                    initializer=tflayers.xavier_initializer(uniform = False))
            tf.summary.histogram(name=self.entity_embedding.op.name, values=self.entity_embedding)
            self.relation_embedding = tf.get_variable(name='relation',
                                                      shape=[self.kg.n_relation, self.embedding_rel_dim],
                                                      initializer=tflayers.xavier_initializer(uniform = False))
            tf.summary.histogram(name=self.relation_embedding.op.name, values=self.relation_embedding)
            self.transfer_matrix = tf.get_variable(name='normal_vector',
                                                      shape=[self.kg.n_relation, self.embedding_ent_dim * self.embedding_rel_dim],
                                                      initializer=tflayers.xavier_initializer(uniform = False))
            tf.summary.histogram(name=self.transfer_matrix.op.name, values=self.transfer_matrix)
        with tf.name_scope('normalization'):
            self.entity_embedding = tf.nn.l2_normalize(self.entity_embedding, axis=1)
            self.relation_embedding = tf.nn.l2_normalize(self.relation_embedding, axis=1)
            self.transfer_matrix = tf.nn.l2_normalize(self.transfer_matrix, axis=1)
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
    
    def __transfer(self, matrix, e):
        return tf.reshape(tf.matmul(matrix, e), [-1, self.embedding_rel_dim])
    
    def __transfer_(self, matrix, e):
        return tf.reshape(tf.matmul(matrix, e), [self.embedding_rel_dim])

    def infer(self, triple_pos, triple_neg):
        with tf.name_scope('lookup'):
            head_pos = tf.reshape(tf.nn.embedding_lookup(self.entity_embedding, triple_pos[:, 0]),[-1, self.embedding_ent_dim, 1])
            tail_pos = tf.reshape(tf.nn.embedding_lookup(self.entity_embedding, triple_pos[:, 1]),[-1, self.embedding_ent_dim, 1])
            relation_pos = tf.reshape(tf.nn.embedding_lookup(self.relation_embedding, triple_pos[:, 2]),[-1, self.embedding_rel_dim, 1])
            matrix_pos = tf.reshape(tf.nn.embedding_lookup(self.transfer_matrix, triple_pos[:, 2]),[-1, self.embedding_rel_dim, self.embedding_ent_dim])

            head_neg = tf.reshape(tf.nn.embedding_lookup(self.entity_embedding, triple_neg[:, 0]),[-1, self.embedding_ent_dim, 1])
            tail_neg = tf.reshape(tf.nn.embedding_lookup(self.entity_embedding, triple_neg[:, 1]),[-1, self.embedding_ent_dim, 1])
            relation_neg = tf.reshape(tf.nn.embedding_lookup(self.relation_embedding, triple_neg[:, 2]),[-1, self.embedding_rel_dim, 1])
            matrix_neg = tf.reshape(tf.nn.embedding_lookup(self.transfer_matrix, triple_neg[:, 2]),[-1, self.embedding_rel_dim, self.embedding_ent_dim])
        with tf.name_scope('link'):
            distance_pos = self.__transfer(matrix_pos, head_pos) + relation_pos - self.__transfer(matrix_pos, tail_pos)
            distance_neg = self.__transfer(matrix_neg, head_neg) + relation_neg - self.__transfer(matrix_neg, tail_neg)
        return distance_pos, distance_neg

    def evaluate(self, eval_triple):
        with tf.name_scope('lookup'):
            head = tf.reshape(tf.nn.embedding_lookup(self.entity_embedding, eval_triple[0]),[self.embedding_ent_dim, 1])
            tail = tf.reshape(tf.nn.embedding_lookup(self.entity_embedding, eval_triple[1]),[self.embedding_ent_dim, 1])
            relation = tf.reshape(tf.nn.embedding_lookup(self.relation_embedding, eval_triple[2]),[self.embedding_rel_dim])
            normal_matrix = tf.reshape(tf.nn.embedding_lookup(self.transfer_matrix, eval_triple[2]),[ self.embedding_rel_dim, self.embedding_ent_dim])
        with tf.name_scope('link'):
            embedding = self.entity_embedding
            distance_head_prediction = []
            distance_tail_prediction = []
            for i in range(self.kg.n_entity):
                e = tf.reshape(embedding[i, :], [self.embedding_ent_dim, 1])
                distance_head_prediction.append(tf.reshape(self.__transfer_(normal_matrix, e) + relation - self.__transfer_(normal_matrix, tail), [1, self.embedding_rel_dim]))
                distance_tail_prediction.append(tf.reshape(self.__transfer_(normal_matrix, head) + relation - self.__transfer_(normal_matrix, e), [1, self.embedding_rel_dim]))
            distance_head_prediction = tf.concat(axis=0, values=distance_head_prediction)
            distance_tail_prediction = tf.concat(axis=0, values=distance_tail_prediction)
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
