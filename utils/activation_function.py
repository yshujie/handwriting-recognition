import numpy as np

def sigmoid(x):
    """
        sigmoid(x) -> y

        参数：
            x: 输入信号

        返回：
            sigmoid 函数的输出

        数学公式：
            y = 1 / (1 + exp(-x))
    """
    return 1 / (1 + np.exp(-x))


def indentity_function(x):
    """
        indentity_function(x) -> x

        参数：
            x: 输入信号

        返回：
            输入信号

        数学公式：
            y = x
    """
    return x