# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
from utils import utils

batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

# 定义我们的模型
from mxnet import gluon

# 新建一个空的层次网络
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(10))
    
# 初始化模型参数
net.initialize()