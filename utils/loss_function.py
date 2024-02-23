import numpy as np

def mean_squared_error(y, t):
    """
        均方误差

        :param y: 神经网络的输出
        :param t: 监督数据
        :return: 均方误差

        数学公式：
        E = 1/2 * Σ(yk - tk)^2
    """
    return 0.5 * np.sum((y-t)**2)


def cross_ertropy_error(y, t):
    """
        交叉熵误差

        :param y: 神经网络的输出
        :param t: 监督数据
        :return: 交叉熵误差

        数学公式：
        E = -Σ(tk * log(yk))
    """
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))    