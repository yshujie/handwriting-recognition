class MulLayer:
    """ 
    乘法层    
    """
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        """ 
        乘法的前向传播

        Args:
            x: 输入数据
            y: 输入数据

        Returns:
            out: 输出数据

        数学公式：
            out = x * y
        """
        self.x = x 
        self.y = y
        out = x * y 

        return out 
    
    def backward(self, dout):
        """ 
        乘法的反向传播

        Args:
            dout: 输出数据的导数

        Returns:
            dx: 输入数据x的导数
            dy: 输入数据y的导数

        数学公式：
            dx = dout * y
            dy = dout * x
        """
        
        dx = dout * self.y 
        dy = dout * self.x

        return dx, dy




class AddLayer:
    """
    加法层
    """
    def __init__(self):
        pass

    def forward(self, x, y):
        """ 
        加法的前向传播

        Args:
            x: 输入数据
            y: 输入数据

        Returns:
            out: 输出数据

        数学公式：
            out = x + y
        """
        out = x + y 

        return out

    def backward(self, dout):
        """
        加法的反向传播

        Args:
            dout: 输出数据的导数

        Returns:
            dx: 输入数据x的导数
            dy: 输入数据y的导数

        数学公式：
            dx = dout * 1
            dy = dout * 1
        """
        dx = dout * 1
        dy = dout * 1

        return dx, dy