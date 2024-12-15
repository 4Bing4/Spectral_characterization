import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
import de2000caculate as delab
import pandas as pd
from kan import *
import matplotlib.pyplot as plt
import pandas as pd
import os
import TRCplot as trc

# 指定文件夹路径
folder_path = 'F:/pythonProject/character_final/data'

def base_function(x):
    return x ** 2.2

# 遍历文件夹中的所有CSV文件
for filename in os.listdir(folder_path):

    # 检查是否有可用的 GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.set_default_dtype(torch.float64)  # 设置默认张量数据类型为双精度浮点数
    torch.manual_seed(33)  # 设置随机数种子为33


    class FullyConnectedNetwork(torch.nn.Module):
        def __init__(self):
            super(FullyConnectedNetwork, self).__init__()
            # 初始化参数
            self.kan_r = KAN(width=[1, 1], grid=5, k=3, seed=0,  base_fun=base_function, device=device, grid_range=[0, 1])
            self.kan_g = KAN(width=[1, 1], grid=5, k=3, seed=0,  base_fun=base_function, device=device, grid_range=[0, 1])
            self.kan_b = KAN(width=[1, 1], grid=5, k=3, seed=0,  base_fun=base_function, device=device, grid_range=[0, 1])
            self.fc1 = nn.Linear(input_features, 300)
            self.fc2 = nn .Linear(300, 300)
            self.fc3 = nn.Linear(300, output_features)
            self.activation = nn.Tanh()
            self.output_activation = nn.Sigmoid()

            # 已知的 3x3 转换矩阵
            self.conversion_matrix = nn.Parameter(torch.tensor(np.array(rgb_m), device=device), requires_grad=False)


        def forward(self, x):

            # kan
            r = x[:, 0]
            g = x[:, 1]
            b = x[:, 2]
            r = r.unsqueeze(1)
            g = g.unsqueeze(1)
            b = b.unsqueeze(1)
            r = self.kan_r(r)
            # self.kan_r.plot()
            g = self.kan_g(g)
            b = self.kan_b(b)

            corrected = torch.stack([r, g, b], dim=1)
            corrected = corrected.squeeze()
            Y_base = torch.matmul(corrected, self.conversion_matrix)

            # 神经网络部分
            x = self.activation(self.fc1(x))
            x = self.activation(self.fc2(x))
            x = self.output_activation(self.fc3(x))

            x = Y_base + 1000 * x - 500
            return x


    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        # 读取CSV文件
        data = pd.read_csv(file_path, skiprows=0)
        # 选择第11列及后续数据
        data = data.iloc[:, 9:]  # iloc是基于0索引的
        data = data.to_numpy()
        # data0 = data[0, :]
        # data = data - data0

    training_step = 35001
    name = filename
    batch_size = 64

    # 选择特定行和列,读取数据并将它们移动到指定设备
    rows_train = np.r_[0, 4, 20, 100, 124:248]  # 第1，125：248行
    x_signals = data[np.ix_(rows_train, np.r_[0:3])]
    y_lut = data[np.ix_(rows_train, np.r_[5:8])]

    rows_test = np.r_[326:526]
    rgbtst = data[np.ix_(rows_test, np.r_[0:3])]
    pcagt = data[np.ix_(rows_test, np.r_[5:8])]
    rgb_m = data[np.ix_(np.r_[20, 4, 100], np.r_[5:8])]

    xyzs = y_lut
    xyzgt = pcagt

    # 初始化
    torch.manual_seed(10)
    input_features = 3
    output_features = 3

    xdata = np.append(x_signals, rgbtst, axis=0)
    target = xyzs

    # 特征归一化
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(xdata)
    xdata = min_max_scaler.transform(xdata)

    train_size = np.size(x_signals, 0)
    x_train, y_train = xdata[0:train_size, :], target[:, :]
    indices = np.random.permutation(x_train.shape[0])
    x_train = x_train[indices]
    y_train = y_train[indices]
    x_tst = xdata[train_size:, :]
    y_tst = xyzgt

    # 定义模型并将其移动到指定设备
    model = FullyConnectedNetwork().to(device)

    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # 定义目标损失函数
    loss_func = torch.nn.MSELoss()

    val_loss_init = 1000000
    recent_losses = []
    tst_loss_list = []

    for step in range(training_step):
        M_train = len(x_train)
        with tqdm(np.arange(0, M_train, batch_size), desc=f'{step}Training...') as tbar:
            for index in tbar:
                L = index
                R = min(M_train, index + batch_size)

                # 训练内容
                train_pre = model(torch.from_numpy(x_train[L:R, :]).to(device))
                train_gt = torch.from_numpy(y_train[L:R, :].reshape(R - L, output_features)).to(device)
                train_loss = loss_func(train_pre, train_gt)


                tst_pre = model(torch.from_numpy(x_tst).to(device))
                tst_gt = torch.from_numpy(y_tst.reshape(len(y_tst), output_features)).to(device)
                tst_loss = loss_func(tst_pre, tst_gt)


                # 保存每次迭代的tst_loss
                tst_loss_list.append(float(tst_loss.data))

                tbar.set_postfix(train_loss=float(train_loss.data),
                                 tst_loss=float(tst_loss.data))
                tbar.update()

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

            recent_losses.append(float(tst_loss.data))
            if len(recent_losses) > 15000:
                recent_losses.pop(0)

            if len(recent_losses) == 15000 and tst_loss > recent_losses[0]:
                break

            if tst_loss < val_loss_init:
                val_loss_init = tst_loss
                best_epoch = step
                torch.save(model, f'result_xyz\\{name}_xyz_best.pt')
                tst_loss_cpu = tst_loss.cpu().detach().numpy()
                # 保存结果
                np.savetxt(f'result_xyz\\{name}_xyz_result.txt',
                           [best_epoch, tst_loss_cpu])
                plot_num = 1

            aa = step % 5000 == 0
            ab = plot_num == 1
            ac = aa & ab
            if ac:
                # plt.clf()
                inputs = model.kan_b.spline_preacts[0][:, 0, 0].cpu().numpy()
                outputs = model.kan_b.spline_postacts[0][:, 0, 0].cpu().numpy()
                rank = np.argsort(inputs)
                inputs = inputs[rank]
                outputs = outputs[rank]
                outputs -= np.min(outputs)
                outputs /= np.max(outputs)
                plt.plot(inputs, outputs, linewidth=2.2, label=f'Step {step}')
                plot_num = 0

    trc.trc(data)
    plt.legend()
    plt.savefig(f'figure\\{name}_xyz_trc.png', dpi=600)
    plt.close()


    torch.save(model, f'result_xyz\\{name}_xyz_final.pt')
