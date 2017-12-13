# -*- coding: utf-8 -*-

# 高维线性回归: 即数据的维度远大于样本的数量（200:20），如何防止
#             过拟合（即模型的学习能力太强啦，因为参数多-和数据样本额的维度对应）

from mxnet import ndarray as nd
from mxnet import autograd, gluon

num_train = 20
num_test = 100
num_inputs = 200


# 生成数据集
true_w = nd.ones((num_inputs, 1)) * 0.01
true_b = 0.05

X = nd.random_normal(shape=(num_train + num_test, num_inputs))
y = nd.dot(X, true_w)
y += 0.01 * nd.random_normal(shape=y.shape)

X_train, X_test = X[:num_train, :], X[num_train, :]
y_train, y_test = y[:num_train], y[num_train:]

# 数据迭代器
import random
batch_size = 1
def data_iter(num_example):
    idx = list(range(num_example))
    random.shuffle(idx)

    for i in range(0, num_example, batch_size):
        j = nd.array(idx[i:min(i+batch_size,num_example)])
        yield X.take(j), y.take(j)


# 创建梯度
def get_params():
    w = nd.random_normal(shape=(num_inputs, 1)) * 0.1
    b = nd.zeros((1,))
    for param in (w, b):
        param.attach_grad()
    return (w, b)



# 定义L2正则化项
# 注意：正则项可以加载模型上，也可加在loss项上
#      但是为了保持模型输出的y的意义，还是将正则项放在loss项上比较合适
def L2_penalty(w, b):
    return (w ** 2).sum() + b ** 2

# 定义模型
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 120
import matplotlib.pyplot as plt


def net(X, w, b):
    return nd.dot(X, w) + b

def square_loss(yhat, y):
    return (yhat - y.reshape(yhat.shape)) ** 2

def SGD(params, lr):
    for param in params:
        param[:] =  param -lr * param.grad

# 注意我们是用正则化项来规范学习到的参数，但是loss + 正则项 并不是样本的损失
# 真正的损失其实就是loss部分，所以此函数不会将正则项算进去
def test(params, X, y):
    return square_loss(net(X, *params), y).mean().asscalar()

# 训练
# lambda(lambd)是正则项前面的系数，这是模型训练预先指定的参数（超参数）
def train(lambd):
    epochs = 10
    learning_rate = 0.002
    params = get_params()
    train_loss = []
    test_loss = []

    for e in range(epochs):
        for data, label in data_iter(num_train):
            with autograd.record():
                output = net(data,*params)
                loss = square_loss(output, label) + lambd * L2_penalty(*params)
            loss.backward()

            SGD(params, learning_rate)

        # 记录每次整体数据迭代的loss（train，test）
        train_loss.append(test(params, X_train, y_train))
        test_loss.append((test(params, X_test, y_test)))

    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.legend(['train', 'test'])
    plt.show()
    return 'learned w[:10]:', params[0][:10], 'learned b:', params[1]


# 指定超参数
print train(0)

print train(2)

