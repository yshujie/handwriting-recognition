import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定

import numpy as np
from utils.mnist import load_mnist
from neuralnet.TwoLayerNetBackpropagation import TwoLayerNetBackpropagation

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(
    normalize=True, 
    one_hot_label=True
)

network = TwoLayerNetBackpropagation(
    input_size = 784,
    hidden_size = 50,
    output_size = 10
)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

# 求各个权重的绝对误差的平均值
for key in grad_numerical.keys():
    diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
    print(key + ":" + str(diff))