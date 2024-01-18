def AND_v1(x1, x2):
    """
    brief: 与门
    desc: 使用感知机实现的与门
    x1: 输入1
    x2: 输入2
    w1: 权重1，代表 x1 的重要程度
    w2: 权重2，代表 x2 的重要程度
    theta: 阈值，代表神经元被激活的容易程度

    数学公式：
    y = 0 (w1*x1 + w2*x2 <= theta)
    y = 1 (w1*x1 + w2*x2 > theta)
    """
    w1, w2, theta = 0.5, 0.5, 0.9
    tmp = x1*w1 + x2*w2
    if tmp > theta:
        return True
    else:
        return False    


import numpy as np
def AND_v2(x1, x2):
    """
    brief: 与门
    desc: 使用 numpy 实现感知机的与门
    x: 输入信号
    w: 信号权重，代表每个信号的重要程度
    b: 偏置，代表神经元被激活的容易程度

    数学公式：
    y = 0 (w1*x1 + w2*x2 <= theta)
    y = 1 (w1*x1 + w2*x2 > theta)
    """

    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.9

    tmp = x*w + b
    if tmp > 0:
        return True
    else:
        return False