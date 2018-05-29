#!/usr/bin/python
# -*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import math
import tensorflow as tf

n_relation = 14
embedding_dim = 4
bound = 6 / math.sqrt(embedding_dim)

embedding = tf.get_variable(name='relation',
                            shape=[n_relation, embedding_dim],
                            initializer=tf.random_uniform_initializer(minval=-bound, maxval=bound))

embedding = tf.nn.l2_normalize(embedding, dim=1)

a = tf.nn.embedding_lookup(embedding, tf.constant(1))
b = tf.nn.embedding_lookup(embedding, tf.constant(4))

print(embedding.get_shape())
print(a.get_shape())

c = a * b
print(c.get_shape())

# d = tf.reduce_sum(c, 1, keep_dims=True)
d = a - tf.reduce_sum(c) * b
print(d.get_shape())

e = embedding * b

embedding = tf.reshape(embedding, [1, 14, 4])
b = tf.reshape(b, [1, 4, 1])
f = tf.matmul(embedding, b)

h = tf.reshape(f, [14, 1])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    em1, a1, b1, c1, d1, e1, f1, h1 = sess.run(fetches=[embedding, a, b, c, d, e, f, h])
    print('embedding')
    print(em1)
    print('a')
    print(a1)
    print('b')
    print(b1)
    print('c')
    print(c1)
    print('d')
    print(d1)
    print('e')
    print(e1)
    print('f')
    print(f1)
    print('h')
    print(h1)
