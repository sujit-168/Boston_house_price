import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 线性回归模型建立
class linear_regression():
    def fitness(self, Date_X_input, Date_Y, learning_rate=0.5, lamda=0.03):
        sample_num, property_num = Date_X_input.shape# 获取样本个数、样本的属性个数
        Date_X = np.c_[Date_X_input, np.ones(sample_num)]
        self.theta = np.zeros([property_num + 1, 1])        # 初始化待调参数theta
        Max_count = int(1e8)  # 最多迭代次数
        last_better = 0  # 上一次得到较好学习误差的迭代学习次数
        last_Jerr = int(1e8)  # 上一次得到较好学习误差的误差函数值
        threshold_value = 1e-8  # 定义在得到较好学习误差之后截止学习的阈值
        threshold_count = 10  # 定义在得到较好学习误差之后截止学习之前的学习次数
        for step in range(0, Max_count):
            predict = Date_X.dot(self.theta)            # 预测值
            J_theta = sum((predict - Date_Y) ** 2) / (2 * sample_num)            # 损失函数
            self.theta -= learning_rate * (lamda * self.theta + (Date_X.T.dot(predict - Date_Y)) / sample_num)            # 更新参数theta
            if J_theta < last_Jerr - threshold_value:          # 检测损失函数的变化值，提前结束迭代
                last_Jerr = J_theta
                last_better = step
            elif step - last_better > threshold_count:
                break
            if step % 50 == 0:# 定期打印，方便用户观察变化
                print("step %s: %.6f" % (step, J_theta))
def predicted(self, X_input):
        sample_num = X_input.shape[0]
        X = np.c_[X_input, np.ones(sample_num, )]
        predict = X.dot(self.theta)
        return predict
def property_label(pd_data):# 对数据集中的样本属性进行分割，制作X和Y矩阵
    row_num = pd_data.shape[0]
    column_num = len(pd_data.iloc[0, 0].split())# 行数、列数
    X = np.empty([row_num, column_num - 1])
    Y = np.empty([row_num, 1])
    for i in range(0, row_num):
        row_array = pd_data.iloc[i, 0].split()
        X[i] = np.array(row_array[0:-1])
        Y[i] = np.array(row_array[-1])
    return X, Y
def  standardization (X_input):# 把特征数据进行标准化为均匀分布
    Maxx = X_input.max(axis=0)
    Minx = X_input.min(axis=0)
    X = (X_input - Minx) / (Maxx - Minx)
    return X, Maxx, Minx
if __name__ == "__main__":
    data = pd.read_csv("housing-data.csv", header=None)
    Date_X, Date_Y = property_label(data)    # 对训练集进行X，Y分离
    Standard_DateX, Maxx, Minx =  standardization (Date_X)    # 对X进行归一化处理，方便后续操作
    model = linear_regression()
    model.fitness(Standard_DateX, Date_Y)
    Date_predict = model.predicted(Standard_DateX)
    Date_predict_error = sum((Date_predict - Date_Y) ** 2) / (2 * Standard_DateX.shape[0])
    print("Test error is %d" % (Date_predict_error))
    print(model.theta)
    t = np.arange(len(Date_predict))
    plt.figure(facecolor='w')
    plt.plot(t, Date_Y, 'c-', lw=1.6, label=u'actual value')
    plt.plot(t, Date_predict, 'm-', lw=1.6, label=u'estimated value')
    plt.legend(loc='best')
    plt.title(u'Boston house price', fontsize=18)
    plt.xlabel(u'case id', fontsize=15)
    plt.ylabel(u'house price', fontsize=15)
    plt.grid()
    plt.show()