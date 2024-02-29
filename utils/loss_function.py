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


def cross_entropy_error(y, t):
    """
        交叉熵误差 (mini-batch 版本)

        Args:
            y: 神经网络的输出
            t: 监督数据

        Returns:
            交叉熵误差

        数学公式：
        E = -1/N * ΣΣ(tk * log(yk))
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
