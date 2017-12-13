# -*- coding: utf-8 -*-


from mxnet import gluon
from mxnet import ndarray as nd


# 下载以及转换数据格式
def transform(data, label):
    return data.astype('float32')/255, label.astype('float32')


mnist_train = gluon.data.vision.FashionMNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.FashionMNIST(train=False, transform=transform)

# 画图
import matplotlib.pyplot as plt


def show_images(images):
    n = images.shape[0]
    _, figs = plt.subplots(1, n, figsize=(15, 15))
    for i in range(n):
        figs[i].imshow(images[i].reshape((28, 28)).asnumpy())
        figs[i].axes.get_xaxis().set_visible(False)
        figs[i].axes.get_yaxis().set_visible(False)
    plt.show()


def get_text_labels(label):
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress,', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in label]


data, label = mnist_train[0:9]
show_images(data)
print(get_text_labels(label))


# 数据读取
batch_size = 256
train_data = gluon.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_data = gluon.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

# 初始化模型参数
num_inputs = 28 * 28
num_outputs = 10

W_nd = nd.random_normal(shape=(num_inputs, num_outputs))
b_nd = nd.random_normal(shape=num_outputs)

params = [W_nd, b_nd]

# 为所有参数分配梯度空间
for param in params:
    param.attach_grad()


# 定义概率分布模型
def softmax(x_nd):
    exp = nd.exp(x_nd)
    partition = exp.sum(axis=1, keepdims=True)
    return exp / partition


# 定义模型
def net(x_nd):
    return softmax(nd.dot(x_nd.reshape((-1, num_inputs)), W_nd) + b_nd)


# 定义交叉熵损失函数
def cross_entropy(yhat, y):
    return -nd.pick(nd.log(yhat), y)


# 计算精度
def accuracy(output, label):
    # 对应是否相等，相等为1，不等为0，对所有这些结果求均值
    return nd.mean(output.argmax(axis=1)==label).asscalar()


# 评估模型在这个数据上的精度
def evaluate_accuracy(data_iter, net):
    acc = 0
    for data, label in data_iter:
        output = net(data)
        acc +=  accuracy(output, label)
    return acc / len(data_iter)



# 训练
import sys
sys.path.append('..')
from utils.utils import SGD
from mxnet import autograd

learning_rate = .1

for epoch in range(5):
    train_loss = 0.
    train_acc = 0.

    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = cross_entropy(output, label)
        loss.backward()

        # 将梯度做平均，这样学习速率不会对batch_size那么敏感
        SGD(params, learning_rate/batch_size)


        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output, label)

    test_acc = evaluate_accuracy(test_data, net)

    print ('Epoch %d. Loss: %f, Train acc %f, Test acc %f' %
           (epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))


# 对训练结果进行预测
data, label = mnist_test[0:9]
show_images(data)
print('true labels')
print(get_text_labels(label))

predicated_labels = net(data).argmax(axis=1)
print('predicated labels')
print(get_text_labels(predicated_labels.asnumpy()))
