import numpy as np
from utils.activation_function import sigmoid, indentity_function

def init_network():
    """
        init_network() -> dict

        初始化神经网络的权重和偏置。
            权重为大写字母 W 加层号，例如 W1 代表第一层的权重。
            偏置为消息字母 b 加层号，例如 b1 代表第一层的偏置。

        返回神经网络的配置字典。
    """

    network = {}

    # 第一层的权重和偏置
    network['W1'] = np.array([
        [0.1, 0.3, 0.5],
        [0.2, 0.4, 0.6]
    ])
    network['b1'] = np.array([0.1, 0.2, 0.3])

    # 第二层的权重和偏置
    network['W2'] = np.array([
        [0.1, 0.4],
        [0.2, 0.5],
        [0.3, 0.6]
    ])
    network['b2'] = np.array([0.1, 0.2])

    # 第三层的权重和偏置
    network['W3'] = np.array([
        [0.1, 0.3],
        [0.2, 0.4]
    ])
    network['b3'] = np.array([0.1, 0.2])

    return network

def forward(network, x):
    """ 
        forward(network, x) -> y

        参数：
            network: 神经网络的配置字典
            x: 输入信号

        返回：
            神经网络的输出

        数学公式：
            a = xW + b
            y = sigmoid(a)
            其中，x 为输入信号，W 为权重，b 为偏置，a 为加权和，y 为输出信号。
    """
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1 
    z1 = sigmoid(a1)

    a2 = np.dot(z1, w2) + b2 
    z2 = sigmoid(a2)

    a3 = np.dot(z2, w3) + b3
    y = indentity_function(a3)

    return y