import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 原始数据列如下：instant,dteday,season,yr,mnth,hr,holiday,weekday,workingday,weathersit,temp,atemp,hum,windspeed,casual,registered,cnt
# 引入数据
data_path = 'Bike-Sharing-Dataset/hour.csv'
rides = pd.read_csv(data_path)
rides.head()
# 绘制前十天的数据
rides[:24 * 10].plot(x='dteday', y='cnt')

# 下面是一些分类变量，例如季节、天气、月份。要在我们的模型中包含这些数据，我们需要创建二进制虚拟变量，利用pd.get_dummies实现
#  pd.get_dummies就是对分组数据进行one_hot编码
dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)

# 删掉多余的数据  去除噪声
fields_to_drop = ['instant', 'dteday', 'season', 'weathersit',
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)
data.head()

quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']

# 对每个连续变量标准化，即转换和调整变量，使它们的均值为 0，标准差为 1  以字典形式存储
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    # loc ：格局标签 对每一行数据进行调整
    data.loc[:, each] = (data[each] - mean) / std

# 取最后21天的数据
test_data = data[-21 * 24:]

# 移除最后21的数据
data = data[:-21 * 24]

# 分离特征和输出
target_fields = ['cnt', 'casual', 'registered']
features, targets = data.drop(target_fields, axis=1), data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]
print(targets)

# 分离验证机和训练集
train_features, train_targets = features[:-60 * 24], targets[:-60 * 24]
val_features, val_targets = features[-60 * 24:], targets[-60 * 24:]


# 设置网络
class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # 设置网络节点
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # 根据网络节点初始化权重
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes ** -0.5,
                                                        (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes ** -0.5,
                                                         (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate

        # 设置激活函数
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))  # Replace 0 with your sigmoid calculation.

    # 以sigmoid作为激活函数
    def sigmoid(self, x):
        result = 1 / (1 + np.exp(-x))
        return result

    # 训练函数
    def train(self, features, targets):
        ''' Train the network on batch of features and targets.

            Arguments
            ---------

            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values

        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            # 前向传播计算输出
            hidden_inputs = np.dot(X, self.weights_input_to_hidden)  # signals into hidden layer
            hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer

            # 前向传播计算输出
            final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)  # signals into final output layer
            final_outputs = final_inputs  # signals from final output layer
            # 计算输出误差
            error = y - final_outputs
            output_error_term = error
            # 计算隐藏层误差
            hidden_error = np.dot(self.weights_hidden_to_output, output_error_term)
            hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)

            # 反向传播求w的偏导
            delta_weights_i_h += hidden_error_term * X[:, None]
            # 反向传播求w的偏导
            delta_weights_h_o += output_error_term * hidden_outputs[:, None]

        # 反向传播更新权重
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records  # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records  # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features

            Arguments
            ---------
            features: 1D array of feature values
        '''
        # 计算隐藏层输出
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)  # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer
        # 计算最后层输出
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs  # signals from final output layer
        # TODO: Output layer - Replace these values with the appropriate calculations.
        return final_outputs


# 平方差误差函数
def MSE(y, Y):
    return np.mean((y - Y) ** 2)


import unittest

# 设置输入
inputs = np.array([[0.5, -0.2, 0.1]])
# 设置输出
targets = np.array([[0.4]])
# 设置初始化权重
test_w_i_h = np.array([[0.1, -0.2],
                       [0.4, 0.5],
                       [-0.3, 0.2]])
# 设置初始化权重
test_w_h_o = np.array([[0.3],
                       [-0.1]])


# 单元测试
class TestMethods(unittest.TestCase):

    ##########
    # Unit tests for data loading
    ##########
    # 测试数据
    def test_data_path(self):
        # Test that file path to dataset has been unaltered
        self.assertTrue(data_path.lower() == 'bike-sharing-dataset/hour.csv')

    # 测试数据
    def test_data_loaded(self):
        # Test that data frame loaded
        self.assertTrue(isinstance(rides, pd.DataFrame))

    ##########
    # Unit tests for network functionality
    ##########
    # 测试激活函数
    def test_activation(self):
        network = NeuralNetwork(3, 2, 1, 0.5)
        # Test that the activation function is a sigmoid
        self.assertTrue(np.all(network.activation_function(0.5) == 1 / (1 + np.exp(-0.5))))

    # 测试训练函数
    def test_train(self):
        # Test that weights are updated correctly on training
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()

        network.train(inputs, targets)
        self.assertTrue(np.allclose(network.weights_hidden_to_output,
                                    np.array([[0.37275328],
                                              [-0.03172939]])))
        self.assertTrue(np.allclose(network.weights_input_to_hidden,
                                    np.array([[0.10562014, -0.20185996],
                                              [0.39775194, 0.50074398],
                                              [-0.29887597, 0.19962801]])))

    # 测试运行函数
    def test_run(self):
        # Test correctness of run method
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()

        self.assertTrue(np.allclose(network.run(inputs), 0.09998924))


suite = unittest.TestLoader().loadTestsFromModule(TestMethods())
unittest.TextTestRunner().run(suite)

import sys

### 超参数设置 ###
iterations = 5000
learning_rate = 0.1
hidden_nodes = 10
output_nodes = 5

N_i = train_features.shape[1]
# 初始化网络
network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

losses = {'train': [], 'validation': []}
# 开始训练
for ii in range(iterations):
    # Go through a random batch of 128 records from the training data set
    # 数据批次化处理，一个批次为128个
    batch = np.random.choice(train_features.index, size=128)
    X, y = train_features.ix[batch].values, train_targets.ix[batch]['cnt']

    network.train(X, y)

    # Printing out the training progress
    # 训练误差
    train_loss = MSE(network.run(train_features).T, train_targets['cnt'].values)
    # 验证误差
    val_loss = MSE(network.run(val_features).T, val_targets['cnt'].values)
    # 打印误差
    sys.stdout.write("\rProgress: {:2.1f}".format(100 * ii / float(iterations)) \
                     + "% ... Training loss: " + str(train_loss)[:5] \
                     + " ... Validation loss: " + str(val_loss)[:5])
    sys.stdout.flush()

    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)

# 绘制误差图和预测未来的共享单车注册趋势
plt.plot(losses['train'], label='Training loss')
plt.plot(losses['validation'], label='Validation loss')
plt.legend()
_ = plt.ylim()

fig, ax = plt.subplots(figsize=(8, 4))

mean, std = scaled_features['cnt']
predictions = network.run(test_features).T * std + mean
ax.plot(predictions[0], label='Prediction')
ax.plot((test_targets['cnt'] * std + mean).values, label='Data')
ax.set_xlim(right=len(predictions))
ax.legend()

dates = pd.to_datetime(rides.ix[test_data.index]['dteday'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24], rotation=45)
