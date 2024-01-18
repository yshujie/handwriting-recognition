# 与门
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.9
    tmp = x1*w1 + x2*w2
    if tmp > theta:
        return True
    else:
        return False