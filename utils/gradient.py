import numpy as np 

def numberical_gradient(f, x):
    """ 
    求梯度

    Args:
        f: 函数
        x: 自变量

    Returns:
        梯度
    """
    h = 1e-4 # 0.0001 
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val # 还原值
        it.iternext()

    return grad