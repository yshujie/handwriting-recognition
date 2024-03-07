import sys, os
sys.path.append(os.pardir) 
import numpy as np
from utils.activation_function import sigmoid, softmax
from utils.loss_function import cross_entropy_error
from utils.gradient import numberical_gradient
from utils.layers import *
from collections import OrderedDict

class TwoLayerNetV1:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """ 
        初始化函数

        Args:
            input_size: 输入层的神经元数
            hidden_size: 隐藏层的神经元数
            output_size: 输出层的神经元数
            weight_init_std: 指定权重的标准差
        """

        # 初始化权重
        self.params = {}

        # 第一层的权重和偏置 
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)

        # 第二层的权重和偏置
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        """ 
        预测函数

        Args:
            x: 输入数据

        Returns:
            y: 预测结果
        """
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1 
        z1 = sigmoid(a1)

        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    def loss(self, x, t):
        """ 
        损失函数

        Args:
            x: 输入数据
            t: 监督数据

        Returns:
            损失值
        """
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        """
        计算精度

        Args:
            x: 输入数据
            t: 监督数据

        Returns:
            精度
        """
        y = self.predict(x)

        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """
        计算权重参数的梯度

        Args:
            x: 输入数据
            t: 监督数据

        Returns:
            梯度
        """
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numberical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numberical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numberical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numberical_gradient(loss_W, self.params['b2'])

        return grads


class TwoLayerNetV2:
    def __init__(
        self, 
        input_size, 
        hidden_size,
        out_put_size,
        weight_init_std=0.01
    ):
        """
        初始化函数

        Args:
            input_size: 输入层的神经元数
            hidden_size: 隐藏层的神经元数
            output_size: 输出层的神经元数
            weight_init_std: 指定权重的标准差
        """

        # 初始化权重
        self.params = {}
        
        self.params['W1'] = weight_init_std * np,random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)

        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # 生成层
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Sigmoid1'] = Sigmoid()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        """ 
        预测函数

        Args:
            x: 输入数据

        Returns:
            y: 预测结果
        """

        for layers in self.layers.values():
            x = layer.forward(x)
        
        return x

    def loss(self, x, t):
        """ 
        损失函数

        Args:
            x: 输入数据
            t: 监督数据

        Returns:
            损失值
        """
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        """ 
        计算精度

        Args:
            x: 输入数据
            t: 监督数据

        Returns:
            精度
        """

        y = self.predict(x)
        y = np.argmax(y, axis=1)

        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])

        return accuracy

    def numerical_gradient(self, x, t):
        """
        计算权重参数的梯度

        Args:
            x: 输入数据
            t: 监督数据

        Returns:
            梯度
        """
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numberical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numberical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numberical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numberical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):
        """ 
        计算权重参数的梯度

        Args:
            x: 输入数据
            t: 监督数据

        Returns:
            梯度
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads    