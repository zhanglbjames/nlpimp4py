# -*- coding: utf-8 -*-

# 导入ndArray数组矩阵工具
from mxnet import ndarray as nd
# 导入自动求导工具
from mxnet import autograd


# 记录模型输入特征的数量
num_features =2

# 记录数据的数量
num_examples = 1000

# 模拟的正确的参数
true_w = [2,-3.4]
true_b = 5.0

# 生成原始数据集(均值为0，方差为1的正太分布)
origin_data = nd.random_normal(shape=(num_examples,num_features))

# 对原始数据加噪声,噪声是正太分布，得到label的值
# y_hat = w*x + b + noise
y_hat = true_w[0]*origin_data[:, 0] + true_w[1]*origin_data[:, 1] + true_b
y_hat += 0.01 * nd.random_normal(shape=y_hat.shape)

import matplotlib.pyplot as plt
plt.scatter(origin_data[:, 1].asnumpy(),y_hat.asnumpy())
plt.show()

import random
batch_size = 10


def data_iter():
    # 产生一个随机索引列表
    idx = list(range(num_examples))
    # 将索引列表打乱
    random.shuffle(idx)
    # start-end step-size
    for i in range(0,num_examples,batch_size):
        # idx的下标对应的值是origin_data数组的下标
        j = nd.array(idx[i:min(i+batch_size,num_examples)])
        # 一次性取出batch-size的数据以及label值
        yield nd.take(origin_data,j),nd.take(y_hat,j)


for data, label in data_iter():
    print (data, label)
    break

# 定义待估计的W参数数组
# 参数的初始化是一些归一化的随机值，但是实际上参数的初始值是影响模型的训练的
w = nd.random_normal(shape=(num_features,1))
b = nd.zeros((1,))
params = [w,b]


# 创建梯度
# 对参数记录其梯度，用来计算和这些参数相关的损失函数的梯度的大小
for param in params:
    param.attach_grad()


# 定义模型
def net(x):
    return nd.dot(x,w) + b


# 定义损失函数
def square_loss(yhat,y):
    # 将y变换成yhat的形状来避免矩阵形状的自动转换
    return (yhat - y.reshape(yhat.shape)) ** 2


# 定义随机梯度下降的参数按照梯度进行步进的函数
def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

# 定义真实的数据对应的函数，用来画图做对比的
def real_fn(x):
    return 2 * x[:,0] - 3.4 * x[:,1] + 5.0


def plot(losses, x, sample_size=100):
    xs = list(range(len(losses)))
    f, (fg1, fg2) = plt.subplots(1,2)
    fg1.set_title('Loss during training')
    fg1.plot(xs, losses, '-r')
    fg2.set_title('Estimated vs real function')
    fg2.plot(x[:sample_size, 1].asnumpy(),
             net(x[:sample_size, :]).asnumpy(), 'or', label='Estimated')
    fg2.plot(x[:sample_size, 1].asnumpy(),
             real_fn(x[:sample_size, :]).asnumpy(), '*g', label='Real')
    fg2.legend()
    plt.show()


epochs = 5
learning_rate = .001
niter = 0
losses = []
moving_loss = 0
smoothing_constant = .01

# 训练
for e in range(epochs):
    total_loss = 0

    '''
        1. 每个迭代周期都record计算轨迹（loss），计算loss的backward，
           然后每一次都用params的梯度更新params；更新完毕再用新的params
           重新计算loss，然后在对loss进行backward.
           
        2. data_iter一次迭代出来的数据是batch_size（10）大小的
        
        3. loss.backward() 等价于 loss.sum().backward()
           其中loss.sum()就是所有batch_size大小的单行/列直接求和，
           backward在这基础上求和
    '''
    for data, label in data_iter():
        # 记录目标函数（损失）的计算历史轨迹
        with autograd.record():
            output = net(data)
            # loss is ndArray of batch_size * 1
            loss = square_loss(output, label)

        '''
            1. 将loss设为随机梯度下降的优化目标函数，对这个函数的结果进行最优化,
               然后计算之前标记（attach_grad）的变量参数的梯度,注意这步是计算梯度，
               在SGD()中可以直接通过参数来获取已经计算的梯度.
               
            2. 参数得到的梯度 等价于 loss.sum().backward()结果的参数梯度
            
            3. 其实是对batch_size条数据同时求loss，这个loss是多条数据的loss列（shape=[batch_size , 1]）,
               但是SGD的梯度步进step是每一batch_size进行更新一次。
               所以loss列如何得到参数的一个梯度的问题：按理说应该是一条数据对应一个梯度，这里的做法是
               得到的梯度等价于loss.sum().backward()结果的参数梯度，即将loss列的每一行单独求导然后按照对应参数，
               将梯度相加，得到最终的这个batch_size的参数梯度。
               
        '''
        loss.backward()

        '''
            1. 这里用的是sum目标函数进行对模型参数求梯度；
            
            2. 由于sum是线性的，所以这一次对模型参数的更新是
               batch_size中每个数据记录对参数的累加步进量。
               
            3. 这个更新方式和 gluon 的 trainer.step(batch_size)不一样
               它是对batch_size进行归一化（也就是求均值）然后再更新。
        '''
        SGD(params, learning_rate)

        # 统计总的损失和
        total_loss += nd.sum(loss).asscalar()

        # 记录每读取一个数据点后，损失的移动平均值的变化；
        niter +=1
        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss

        # correct the bias from the moving averages
        est_loss = moving_loss/(1-(1-smoothing_constant)**niter)

        if (niter + 1) % 100 == 0:
            losses.append(est_loss)
            print("Epoch %s, batch %s. Moving avg of loss: %s. Average loss: %f" % (e, niter, est_loss, total_loss/num_examples))
            #plot(losses, origin_data)

print true_w, w, true_b, b