#!/usr/bin/python
# -*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import tensorflow as tf
import numpy as np
import math

from model import Model

class TransE(Model):
    def __init__(self, kg, embedding_dim, margin_value, score_func,
                 batch_size, learning_rate, n_generator, n_rank_calculator):
        super(TransE, self).__init__(kg, embedding_dim, margin_value, score_func,
                                     batch_size, learning_rate, n_generator,
                                     n_rank_calculator)