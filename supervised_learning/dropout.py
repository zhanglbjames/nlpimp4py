# -*- coding: utf-8 -*-

# 丢弃法

from mxnet import nd


def dropout(X, drop_probability):
    keep_probability = 1 - drop_probability
    assert 0 <= keep_probability <= 1

    if keep_probability == 0:
        return X.zeros_like()

    # uniform生成 0 ～ 1均匀分布的数值（是否保留的概率），
    # 然后在于keep_probability（概率阀值）进行比较，如果为true则当前位置的元素为1，否则为0
    mask = nd.random_uniform(0, 1.0, X.shape, ctx=X.context) < keep_probability

    # 伸缩变换
    scale = 1 / keep_probability

    # 最后返回mask之后的X，注意除此之外数值的范围进行伸缩变换
    return mask * X * scale


# test dropout
A = nd.arange(20).reshape((5,4))

print(dropout(A, 0.0))
print(dropout(A, 0.5))
print(dropout(A, 1.0))

import sys
sys.path.append('..')
from utils import utils
batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

# 含有两个隐含层的多层感知机
num_inputs = 28 * 28
num_outputs = 10

num_hidden1 = 256
num_hidden2 = 256
weight_scale = 0.1
# scale 为分布的标准差
W1 = nd.random_normal(shape=(num_inputs, num_hidden1),scale=weight_scale)
b1 = nd.zeros(num_hidden1)

W2 = nd.random_normal(shape=(num_hidden1, num_hidden2), scale=weight_scale)
b2 = nd.zeros(num_hidden2)

W3 = nd.random_normal(shape=(num_hidden2, num_outputs), scale=weight_scale)
b3 = nd.zeros(num_outputs)


params = [W1, b1, W2, b2, W3, b3]

for param in params:
    param.attach_grad()


# 在激活函数后添加丢弃层

drop_prob1 = 0.2
drop_prob2 = 0.5

'''
    修正： 一种说法是：在做training的时候用dropout；在做testing的估计的时候，
        不能用dropout，要用完整的structure，但是代码里没有看出这一点，
        在testing的时候还是用了dropout后的网络。
    
    note：这里使用了autograd.is_trainning()的方法，默认被 with autograd.record()的部分都是training状态的
         即当函数处于这个作用域范围会被标记为training状态。
'''
def net(X):
    X = X.reshape((-1, num_inputs))

    # 第一层全连接
    hidden1 = nd.relu(nd.dot(X, W1) + b1)
    # 在第一层之后添加dropout层
    if autograd.is_training():
        print("#")
        hidden1 = dropout(hidden1, drop_prob1)
    else:
        print("+")

    # 第二层全连接
    hidden2 = nd.relu(nd.dot(hidden1, W2) + b2)
    if autograd.is_training():
        # 在第二层之后添加dropout层
        print("/")
        hidden2 = dropout(hidden2, drop_prob2)
    else:
        print("-")
    return nd.dot(hidden2, W3) + b3


# 训练
'''
    推荐把更靠近输出层的元素的丢弃率设置的更小一点
'''
from mxnet import autograd, gluon

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

learning_rate = 0.5

for epoch in range(5):
    train_loss = 0.0
    train_acc = 0.0
    print("\ntrain")
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        utils.SGD(params, learning_rate/batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output, label)

    print("\ntest")
    test_acc = utils.evaluate_accuracy(test_data, net)

    # 打印
    utils.print_info(epoch, train_loss, train_acc, test_acc, len(train_data))





