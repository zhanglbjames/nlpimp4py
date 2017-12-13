# -*- coding: utf-8 -*-

# 多项式拟合测试
# y = 1.2x - 3.4x**2 + 5.6x**3 + 5.0 + noise
# 创建数据集
from mxnet import ndarray as nd
from mxnet import autograd, gluon

num_train = 100
num_test = 100
true_w = [1.2, -3.4, 5.6]
true_b = 5.0

# 生成数据集
x = nd.random_normal(shape=(num_train + num_test, 1))
X = nd.concat(x, nd.power(x,2), nd.power(x, 3))
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_w[2] * X[:, 2] + true_b
y += .1 * nd.random_normal(shape=y.shape)

import matplotlib as mpl
mpl.rcParams['figure.dpi']= 120
import matplotlib.pyplot as plt


def train(X_train, X_test, y_train, y_test):
    # 线性回归模型
    # gluon 自动推断输入层节点的数量
    #   1. 如果是x的话就是一个节点，那么就有一个权重参数和一个偏移参数
    #   2. 如果是X的话，因为X是N行3列的矩阵，所有会有三个权重和一个便宜参数
    # gluon需要显示的指定输出节点的数量（这里为1）
    net = gluon.nn.Sequential()
    with net.name_scope():
        # 设置输出层的大小
        net.add(gluon.nn.Dense(1))
    net.initialize()
    # 设一些默认参数
    learning_rate = 0.01
    epochs = 100
    batch_size = min(10, y_train.shape[0])
    dataset_train = gluon.data.ArrayDataset(X_train, y_train)
    data_iter_train = gluon.data.DataLoader(
        dataset_train, batch_size, shuffle=True)
    # 默认SGD和均方误差
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {
        'learning_rate': learning_rate})
    square_loss = gluon.loss.L2Loss()
    # 保存训练和测试损失
    train_loss = []
    test_loss = []

    for e in range(epochs):
        for data, label in data_iter_train:
            with autograd.record():
                output = net(data)
                loss = square_loss(output, label)
            loss.backward()
            trainer.step(batch_size)
        train_loss.append(square_loss(
            net(X_train), y_train).mean().asscalar())
        test_loss.append(square_loss(
            net(X_test), y_test).mean().asscalar())
    # 打印结果
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.legend(['train','test'])
    plt.show()
    return ('learned weight', net[0].weight.data(),
            'learned bias', net[0].bias.data())


# 三项式拟合
train(X[:num_train, :], X[num_train:, :], y[:num_train], y[num_train:])
# 线性拟合
train(x[:num_train, :], x[num_train:, :], y[:num_train], y[num_train:])
