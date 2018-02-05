# -*- coding: utf-8 -*-

from mxnet import ndarray as nd
from mxnet import autograd, gluon


num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

X = nd.random_normal(shape=(num_examples, num_inputs))
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
y += .01 * nd.random_normal(shape=y.shape)

batch_size = 10
dataset = gluon.data.ArrayDataset(X,y)
data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True)

for data, label in data_iter:
    print (data,label)

# 定义空的模型
# Sequential指的是相继的网络
net = gluon.nn.Sequential()

# Dense(k)定义的是添加输出节点数量为k个的Dense层（多交织层）
net.add(gluon.nn.Dense(1))

# 初始化模型参数
net.initialize()

# 指定损失函数 L2范数损失
square_loss = gluon.loss.L2Loss()

# 定义一个训练器实例，然后将参数传递给他，进行训练
trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate': 0.1})


# 迭代进行训练

epochs = 10
batch_size = 10
for e in range(epochs):
    total_loss = 0
    for data, label in data_iter:
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)

        loss.backward()

        '''
            1. 相当于把计算出来的梯度除以batch size，因为loss.backward()相当于nd.sum(loss).backward()，
               也就是把一个batch的loss都加起来求的梯度，所以通过除以batch size能够弱化batch size在更新参数时候的影响。
               
            2. 相当于对loss除以10之后求导更新参数，所以前一节的梯度是这一节的梯度的10倍，
               所以在本节中lr增大10倍，从而达到相同的结果。
               可以通过打印前后两节的梯度观察这一结果
        '''
        trainer.step(batch_size)

        total_loss += nd.sum(loss).asscalar()

    print ("Epoch %d, average loss: %f" % (e, total_loss/num_examples))


# 验证
dense = net[0]
# 打印权重
print(true_w, dense.weight.data())
# 打印位移
print(true_b, dense.bias.data())
