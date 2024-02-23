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