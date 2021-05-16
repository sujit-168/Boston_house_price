#此处第一个评价指标使用的是负mae，越大说明模型效果越好;第二个评价指标是r2_score，越大说明模型效果越好。
#无论是对全数据集还是仅提取部分特征的数据集，
#对于图一(负mae):GradientBoost效果最佳，RandomForest、ExtraTrees紧随其后；
#对于图二(r2_score):GradientBoost效果最佳，RandomForest、ExtraTrees紧随其后；

#逐步调参后的多模型效果比较
#思路：对每个模型进行调参，在每个模型达到最优效果时整体比较所有模型
#1.gridsearchcv遍历每个模型的主要参数，求出每个模型的最佳参数。
#2.分别求出feature=3(只考虑'RM', 'PTRATIO', 'LSTAT' 3个features)与feature=13(考虑所有features)时各模型在最佳参数下的r2_Score效果，得出feature=3与13时各自的最佳模型。
#3.比较feature=3与13时各自最佳模型的r2_Score平均值与方差，得出全局最优模型。

import numpy as np
import json


# 读入数据的函数
def load_data():
    data_file = 'bostonh.csv'
    # 从文件读入数据，并指定分隔符为空格
    data = np.fromfile(data_file, sep=' ')
    # 此时data.shape为(7084,)

    # 每条数据包含14项，前13项为影响因素，第14项为价格的中位数
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                     'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    feature_num = len(feature_names)

    # 将原始数据进行reshape， 变成[n,14]的形状
    data = data.reshape([data.shape[0] // feature_num, feature_num])
    # 于是，data.shape变成了(506, 14)

    # 将数据集拆分为训练集和测试集
    # 这里使用80%为训练集， 20%为测试集
    # 训练集和测试集必须没有交集

    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[: offset]


    # 计算训练集的最大值、最小值、平均值     形状为（14,）
    maximums = training_data.max(axis=0)
    minimums = training_data.min(axis=0)
    avgs = training_data.sum(axis=0) / training_data.shape[0]

    # 对数据进行归一化
    for i in range(feature_num):
        data[:, i] = (data[:, i] - minimums[i]) / (maximums[i] - minimums[i])

    training_data = data[: offset]
    test_data = data[offset:]
    return training_data, test_data


class NetWork(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 线性回归模型
        # w的形状是(13, 1)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0

    # 正向传播
    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return z

    # 均方误差损失函数
    def loss(self, z, y):
        error = z-y
        num_samples = error.shape[0]
        cost = error * error
        # 把所有样本的cost相加，求平均
        cost = np.sum(cost)/num_samples
        return cost

    # 梯度
    def gradient(self, x, y):
        z = self.forward(x)
        gradient_w = (z-y) * x
        gradient_w = np.mean(gradient_w, axis=0)
        gradient_w = gradient_w[:, np.newaxis]

        gradient_b = z-y
        gradient_b = np.mean(gradient_b)
        return gradient_w, gradient_b

    # 更新梯度
    def update(self, gradient_w, gradient_b, eta=0.01):
        self.w -= eta * gradient_w
        self.b -= eta * gradient_b

    # 训练函数
    def train(self, training_data, num_epochs, batch_size=10, eta=0.01):
        losses = []
        n = len(training_data)
        for epoch_id in range(num_epochs):
            # 在每轮迭代开始之前，将训练数据的顺序随机打乱
            # 然后再按每次取batch_size条数据的方式取出
            np.random.shuffle(training_data)

            # 将训练数据拆分
            mini_batches = [training_data[k: k+ batch_size] for k in range(0, n, batch_size)]

            for iter_id, mini_batch in enumerate(mini_batches):
                x = mini_batch[:, :-1]
                y = mini_batch[:, -1:]

                a = self.forward(x)
                L = self.loss(a, y)
                gradient_w, gradient_b = self.gradient(x, y)
                self.update(gradient_w, gradient_b, eta)
                losses.append(L)
                print('Epoch {:3d} / iter {:3d}, loss = {:.4f}'.
                      format(epoch_id, iter_id, L))

        return losses


if __name__ == '__main__':
    traing_data, test_data = load_data()
    print(traing_data)
    # x的形状是(404, 13)
    # y的形状是(404, 1)
    x = traing_data[:, : -1]
    y = traing_data[:, -1:]

    net = NetWork(13)
    num_epoches = 10000
    losses = net.train(traing_data, num_epochs=num_epoches, batch_size=100, eta=0.01)
    # 训练结果可视化
    import matplotlib.pyplot as plt
    plot_x = np.arange(len(losses))
    plot_y = np.array(losses)
    plt.plot(plot_x, plot_y)
    plt.show()