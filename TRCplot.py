import numpy as np
import matplotlib.pyplot as plt

def trc(x):
    data = x[np.ix_(np.r_[0:526], np.r_[0:3,4:7])]

    # 提取满足条件的行，然后从中提取所需的列
    t1_lr = data[np.logical_and(data[:, 1] == data[:, 2], data[:, 1] == 0)]
    t1_lg = data[np.logical_and(data[:, 0] == data[:, 2], data[:, 0] == 0)]
    t1_lb = data[np.logical_and(data[:, 0] == data[:, 1], data[:, 0] == 0)]

    # 只提取第1列和第5列 (索引 0 和 4)
    t1_lr = t1_lr[:, [0, 4]]
    t1_lg = t1_lg[:, [1, 4]]
    t1_lb = t1_lb[:, [2, 4]]

    # 根据 t1_lr[:, 0]（输入信号）的大小进行排序
    t1_lr = t1_lr[np.argsort(t1_lr[:, 0])]
    t1_lg = t1_lg[np.argsort(t1_lg[:, 0])]
    t1_lb = t1_lb[np.argsort(t1_lb[:, 0])]

    # 归一化数据（仅对亮度值进行归一化，也就是对第2列）
    t1_lr[:, 1] /= np.max(t1_lr[:, 1])
    t1_lg[:, 1] /= np.max(t1_lg[:, 1])
    t1_lb[:, 1] -= np.min(t1_lb[:, 1])
    t1_lb[:, 1] /= np.max(t1_lb[:, 1])

    # 绘制 TRC 曲线
    # plt.plot(t1_lr[:, 0] / 255, t1_lr[:, 1], label='r')
    plt.plot(t1_lg[:, 0] / 255, t1_lg[:, 1], label='TRC',color='r')
    # plt.plot(t1_lb[:, 0] / 255, t1_lb[:, 1], label='b')

    # plt.legend()
    plt.xlabel('Input signals')
    plt.ylabel('Luminance')
    # plt.grid(True)

    # # 显示图像
    # plt.show()
