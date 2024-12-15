import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
import pandas as pd
from kan import *
import matplotlib.pyplot as plt
import os
import TRCplot as trc


def base_function(x):
    return x ** 2.2

muti = 4000
# 指定文件夹路径
folder_path = 'F:/pythonProject/character_final/data'

# 遍历文件夹中的所有CSV文件
for filename in os.listdir(folder_path):
    # for num_train in range(2):
    # for num_train in [0.99,0.9999,0.99999,0.999995,0.999999,0.99999999]:
    for num_train in [0.9999]:

        for batch_size in [64]:

            # 检查是否有可用的 GPU
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            torch.set_default_dtype(torch.float64)  # 设置默认张量数据类型为双精度浮点数
            torch.manual_seed(33)  # 设置随机数种子为33


            # grid更改在此处
            class FullyConnectedNetwork(torch.nn.Module):
                def __init__(self):
                    super(FullyConnectedNetwork, self).__init__()
                    # 初始化参数
                    # self.kan_r = KAN(width=[1, 1], grid=15, k=3, seed=0, base_fun=lambda x: x ** 2.2, device=device)
                    # self.kan_g = KAN(width=[1, 1], grid=15, k=3, seed=0, base_fun=lambda x: x ** 2.2, device=device)
                    # self.kan_b = KAN(width=[1, 1], grid=15, k=3, seed=0, base_fun=lambda x: x ** 2.2, device=device)
                    self.kan_r = KAN(width=[1, 1], grid=5, k=3, seed=0, base_fun=base_function, device=device,
                                     grid_range=[0, 1])
                    self.kan_g = KAN(width=[1, 1], grid=5, k=3, seed=0, base_fun=base_function, device=device,
                                     grid_range=[0, 1])
                    self.kan_b = KAN(width=[1, 1], grid=5, k=3, seed=0, base_fun=base_function, device=device,
                                     grid_range=[0, 1])
                    self.fc1 = nn.Linear(input_features, 300)
                    self.fc2 = nn.Linear(300, 300)
                    self.fc3 = nn.Linear(300, output_features)
                    self.activation = nn.Tanh()
                    self.output_activation = nn.Sigmoid()

                    # 已知的 3x3 转换矩阵
                    self.conversion_matrix = nn.Parameter(torch.tensor(np.array(rgb_m), device=device),
                                                          requires_grad=False)

                    # self.mutiple = nn.Parameter(torch.tensor(10.0))
                    # self.bias = nn.Parameter(torch.tensor(5.0))

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

                    # 中心化数据
                    Y_base_centered = Y_base - mean

                    # 执行 PCA 转换（投影到主成分）
                    Y_base = torch.matmul(Y_base_centered, components.T)*muti

                    # 神经网络部分
                    x = self.activation(self.fc1(x))
                    x = self.activation(self.fc2(x))
                    x = self.output_activation(self.fc3(x))

                    x = Y_base + 100*x -50
                    # print(self.mutiple.data,self.bias.data)
                    return x


            if filename.endswith('.csv'):
                file_path = os.path.join(folder_path, filename)
                # 读取CSV文件
                data = pd.read_csv(file_path, skiprows=0)
                # 选择第11列及后续数据
                data = data.iloc[:, 9:]  # iloc是基于0索引的
                data = data.to_numpy()

            training_step = 6001
            name = filename

            # 选择特定行和列,读取数据并将它们移动到指定设备
            rows_train = np.r_[0, 4, 20, 100, 124:248]  # 第1，125：248行
            x_signals = data[np.ix_(rows_train, np.r_[0:3])]
            y_lut = data[np.ix_(rows_train, np.r_[27:428])]

            rows_test = np.r_[326:526]
            rgbtst = data[np.ix_(rows_test, np.r_[0:3])]
            pcagt = data[np.ix_(rows_test, np.r_[27:428])]
            rgb_m = data[np.ix_(np.r_[20, 4, 100], np.r_[27:428])]
            cmf = np.loadtxt('ciexyz31_1s.txt')

            # # 标准化
            # scaler = StandardScaler()
            # y_lut = scaler.fit_transform(y_lut)
            # y_lut = scaler.transform()
            # y_lut = scaler.transform(y_lut)

            # 创建 PCA 实例
            pca = PCA()

            # 拟合数据
            pca.fit(y_lut)

            # 获取累计解释方差
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

            # 确定解释 99.99% 方差所需的最小主成分数目
            # num_components = np.argmax(cumulative_variance >= 0.9999) + 1
            num_components = np.argmax(cumulative_variance >= num_train) + 1
            # num_components = 20
            print(f"Number of components needed to explain 99.99% of the variance: {num_components}")

            # 7. 重新创建 PCA 对象，使用选择的主成分数目
            pca = PCA(n_components=num_components)
            pca = pca.fit(y_lut)
            xyzs = pca.transform(y_lut)*muti
            xyzgt = pca.transform(pcagt)*muti

            # 将 PCA 组件和均值转换为 PyTorch 张量
            components = torch.tensor(pca.components_, dtype=torch.float64, device=device)
            mean = torch.tensor(pca.mean_, dtype=torch.float64, device=device)
            cmf = torch.tensor(cmf, dtype=torch.float64, device=device)

            # 初始化
            torch.manual_seed(10)
            input_features = 3
            output_features = num_components

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
                        train_pre_spectrum = torch.matmul(train_pre/muti, components) + mean
                        train_gt_spectrum = torch.matmul(train_gt/muti, components) + mean
                        train_pre_xyz = torch.matmul(train_pre_spectrum, cmf)
                        train_gt_xyz = torch.matmul(train_gt_spectrum, cmf)
                        a1 = torch.mean(torch.sqrt(torch.mean(torch.square(train_pre_xyz - train_gt_xyz), dim=1)))
                        a2 = torch.mean(
                            torch.sqrt(torch.mean(torch.square(train_pre_spectrum - train_gt_spectrum), dim=1)))
                        train_loss = 0 * a1 + 1 * a2

                        tst_pre = model(torch.from_numpy(x_tst).to(device))
                        tst_gt = torch.from_numpy(y_tst.reshape(len(y_tst), output_features)).to(device)
                        tst_loss = loss_func(tst_pre, tst_gt)
                        tst_pre_spectrum = torch.matmul(tst_pre/muti, components) + mean
                        tst_gt_spectrum = torch.matmul(torch.tensor(xyzgt).to(device)/muti, components) + mean
                        tst_pre_xyz = torch.matmul(tst_pre_spectrum, cmf)
                        tst_gt_xyz = torch.matmul(tst_gt_spectrum, cmf)
                        b1 = torch.mean(torch.sqrt(torch.mean(torch.square(tst_pre_xyz - tst_gt_xyz), dim=1)))
                        b2 = torch.mean(torch.sqrt(torch.mean(torch.square(tst_pre_spectrum - tst_gt_spectrum), dim=1)))
                        tst_loss = 0 * b1 + 1 * b2

                        # 保存每次迭代的tst_loss
                        tst_loss_list.append(float(tst_loss.data))

                        tbar.set_postfix(train_loss=float(train_loss.data),
                                         tst_loss=float(tst_loss.data))
                        tbar.update()

                        optimizer.zero_grad()
                        train_loss.backward()
                        optimizer.step()

                    recent_losses.append(float(tst_loss.data))
                    if len(recent_losses) > 5000:
                        recent_losses.pop(0)

                    if len(recent_losses) == 5000 and tst_loss > recent_losses[0]:
                        break

                    if tst_loss < val_loss_init:
                        val_loss_init = tst_loss
                        best_epoch = step
                        torch.save(model, f'result\\{name}_{num_train}_{batch_size}_best.pt')
                        tst_loss_cpu = tst_loss.cpu().detach().numpy()
                        # 保存结果
                        np.savetxt(f'result\\{name}_{num_train}_{batch_size}_result.txt',
                                   [best_epoch, tst_loss_cpu, num_components])

                    if step % 2000 == 0:
                        # plt.clf()
                        # inputs = model.kan_b.spline_preacts[0][:, 0, 0].cpu().numpy()
                        # outputs = model.kan_b.spline_postacts[0][:, 0, 0].cpu().numpy()
                        # rank = np.argsort(inputs)
                        # inputs = inputs[rank]
                        # outputs = outputs[rank]
                        # outputs -= np.min(outputs)
                        # outputs /= np.max(outputs)
                        # plt.plot(inputs, outputs, label=f'B_Step {step}')

                        inputs = model.kan_g.spline_preacts[0][:, 0, 0].cpu().numpy()
                        outputs = model.kan_g.spline_postacts[0][:, 0, 0].cpu().numpy()
                        rank = np.argsort(inputs)
                        inputs = inputs[rank]
                        outputs = outputs[rank]
                        outputs -= np.min(outputs)
                        outputs /= np.max(outputs)
                        plt.plot(inputs, outputs, label=f'f(x)_Step {step}')

                        # inputs = model.kan_r.spline_preacts[0][:, 0, 0].cpu().numpy()
                        # outputs = model.kan_r.spline_postacts[0][:, 0, 0].cpu().numpy()
                        # rank = np.argsort(inputs)
                        # inputs = inputs[rank]
                        # outputs = outputs[rank]
                        # outputs -= np.min(outputs)
                        # outputs /= np.max(outputs)
                        # plt.plot(inputs, outputs, label=f'R_Step {step}')

            trc.trc(data)
            plt.legend(loc='upper left')
            plt.savefig(f'figure\\{name}_{num_train}_{batch_size}_trc.svg', dpi=300, format="svg")
            plt.close()

            # 绘制tst_loss随迭代次数的变化曲线
            plt.figure(2)
            plt.plot(np.arange(len(tst_loss_list)), tst_loss_list, label='Test Loss', color='r', linewidth=2)

            # 添加图例和标签
            plt.title('Test Loss over Iterations')
            plt.xlabel('Iteration')
            plt.ylabel('Test Loss')
            plt.grid(True)
            plt.legend()
            plt.savefig(f'figure\\{name}_{num_train}_{batch_size}_Test Loss.svg', dpi=300, format="svg")
            plt.close()

            torch.save(model, f'result\\{name}_{num_train}_{batch_size}_final.pt')
            # out = model(torch.from_numpy(x_tst).to(device))
            # xyzpre = out.detach().cpu().numpy()
            #
            # # 计算mse
            # pcapre = pca.inverse_transform(xyzpre / 1000)
            # np.savetxt(f'result\\{filename}_pre.txt', pcapre)
            # rmse = np.sqrt(np.mean((pcapre - pcagt) ** 2))
            # print(f"{filename}RMSE: {rmse:.8f}")
