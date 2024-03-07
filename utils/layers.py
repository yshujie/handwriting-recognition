from utils.activation_function import softmax
from utils.loss_function import cross_entropy_error

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        """ 
        ReLU的前向传播

        Args:
            x: 输入数据, 一个numpy数组

        Returns:
            out: 输出数据, 一个numpy数组

        数学公式：
            out = x (x > 0)
            out = 0 (x <= 0)
        """

        # mask 是一个布尔数组
        # 数组的结构和 x 相同
        # 数组的元素是 X 元素与 0 对比后的布尔值
        self.mask = (x <= 0)

        # 创建一个和 x 结构相同的数组
        out = x.copy()
        # 将 mask 为 True 的元素设置为 0
        out[self.mask] = 0

        return out

    def backward(self, dout):
        """ 
        ReLU的反向传播

        Args:
            dout: 输出数据的导数

        Returns:
            dx: 输入数据的导数

        数学公式：
            dx = dout (x > 0)
            dx = 0 (x <= 0)
        """
        # 将 mask 为 True 的元素设置为 0
        dout[self.mask] = 0
        # 返回 dout
        dx = dout

        return dx

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        """ 
        Sigmoid的前向传播

        Args:
            x: 输入数据, 一个numpy数组

        Returns:
            out: 输出数据, 一个numpy数组

        数学公式：
            out = 1 / (1 + exp(-x))
        """
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        """
        Sigmoid的反向传播

        Args:
            dout: 输出数据的导数

        Returns:
            dx: 输入数据的导数

        数学公式：
            dx = dout * (1 - out) * out
        """
        dx = dout * (1.0 - self.out) * self.out

        return dx

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        """ 
        Affine的前向传播

        Args:
            x: 输入数据, 一个numpy数组

        Returns:
            out: 输出数据, 一个numpy数组

        数学公式：
            out = x * W + b
        """
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        """ 
        Affine的反向传播

        Args:
            dout: 输出数据的导数

        Returns:
            dx: 输入数据的导数
            dW: 权重的导数
            db: 偏置的导数

        数学公式：
            dx = dout * W.T
            dW = x.T * dout
            db = np.sum(dout, axis=0)
        """
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 损失
        self.y = None    # softmax的输出
        self.t = None    # 监督数据（one-hot vector）

    def forward(self, x, t):
        """
        SoftmaxWithLoss的前向传播

        Args:
            x: 输入数据
            t: 监督数据

        Returns:
            loss: 损失

        数学公式：
            loss = -1/N * ΣΣ(tk * log(yk))
        """
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        """
        SoftmaxWithLoss的反向传播

        Args:
            dout: 输出数据的导数

        Returns:
            dx: 输入数据的导数

        数学公式：
            dx = (yk - tk) / N
        """
        # 获取 batch_size, 也就是 y 的行数
        batch_size = self.t.shape[0]
        # 如果监督数据是 one-hot-vector, 转换为正确解标签的索引
        dx = (self.y - self.t) / batch_size

        return dx