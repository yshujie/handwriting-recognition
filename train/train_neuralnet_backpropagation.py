import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定

import numpy as np
from utils.mnist import load_mnist
from neuralnet.TwoLayerNetBackpropagation import TwoLayerNetBackpropagation

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(
    normalize=True, 
    one_hot_label=True
)

# 超参数
train_loss_list = []
train_acc_list = []
test_acc_list = []

# 超参数
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

# 平均每个 epoch 的重复次数
iter_per_epoch = max(train_size / batch_size, 1)

network = TwoLayerNetBackpropagation(
    input_size = 784,
    hidden_size = 50,
    output_size = 10
)

for i in range(iters_num):
    print("the ", i, "th iteration is running ...")
    # 获取 mini-batch
    batch_musk = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_musk]
    t_batch = t_train[batch_musk]
    # print("---- x_batch.shape: ", x_batch.shape)

    # 计算梯度，反向传播方式计算梯度
    grad = network.gradient(x_batch, t_batch)
    # print("---- grad['W1'].shape: ", grad['W1'].shape)
    # print("---- grad['b1'].shape: ", grad['b1'].shape)
    # print("---- grad['W2'].shape: ", grad['W2'].shape)
    # print("---- grad['b2'].shape: ", grad['b2'].shape)

    # 更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
        # print("---- network.params[key].shape: ", network.params[key].shape)

    # 计算损失函数的值
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    # print("---- loss: ", loss)

    # 计算每个 epoch 的识别精度
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("---- train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

    print("the ", i, "th iteration is done ...")

print("result:")
print("train_loss_list: ", train_loss_list)
print("train_acc_list: ", train_acc_list)
print("test_acc_list: ", test_acc_list)

print("Done!")