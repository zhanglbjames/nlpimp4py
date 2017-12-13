# -*- coding: utf-8 -*-

'''
    mlp多层感知机：和多类逻辑回归很相似，不过在输出层和输入层多添加了一个或者
    多个隐藏层
'''
import  sys
sys.path.append('..')
from utils import utils

batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

# 定义一个只有一层的隐藏层模型
from mxnet import ndarray as nd

num_inputs = 28 * 28
num_outputs = 10

num_hidden = 255
weight_scale = 0.01

W1 = nd.random_normal(shape=(num_inputs, num_hidden), scale=weight_scale)
b1 = nd.zeros(num_hidden)

W2 = nd.random_normal(shape=(num_hidden, num_outputs),scale=weight_scale)
b2 = nd.zeros(num_outputs)

params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()


# 定义非线性激活函数
def relu(X):
    # 先广播，然后对应的取两者之间的最大值组成返回矩阵
    return nd.maximum(X, 0)

# 定义模型
def net(X):
    # -1表示行的大小自动推断，列的大小为nun_inputs
    X = X.reshape((-1, num_inputs))
    # 隐藏层的数据结果
    hidden1 = relu(nd.dot(X, W1) + b1)
    output = nd.dot(hidden1, W2) + b2
    return output

# 使用gluon中提供的softmax和交叉熵函数，可以保证数值更稳定
from mxnet import gluon

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

# 训练
from mxnet import autograd as autograd

learning_rate = .3

for epoch in range(10):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            # softmax 看来是放在最后的输出层的激活函数了，relu只是隐藏层的激活函数
            loss = softmax_cross_entropy(output, label)
        loss.backward()

        utils.SGD(params, learning_rate/batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output, label)

    test_acc = utils.evaluate_accuracy(test_data, net)

    print('Epoch %d. Loss: %f, Train acc: %f, Test acc %f' %
          (epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))


