def AND_v1(x1, x2):
    """
    brief: 与门
    desc: 使用感知机实现的与门
    x1: 输入1
    x2: 输入2
    w1: 权重1，代表 x1 的重要程度
    w2: 权重2，代表 x2 的重要程度
    theta: 阈值，代表神经元被激活的容易程度
    """
    w1, w2, theta = 0.5, 0.5, 0.9
    tmp = x1*w1 + x2*w2
    if tmp > theta:
        return True
    else:
        return False    