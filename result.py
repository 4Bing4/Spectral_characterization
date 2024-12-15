from kan import *
import torch
import pandas as pd
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import time


def base_function(x):
    return x ** 2.2

muti = 4000

result_path = 'F:/pythonProject/character_final/result_learning0.0001'
# 指定文件夹路径
folder_path = 'F:/pythonProject/character_final/data'

t = []

# 遍历文件夹中的所有CSV文件
for filename in os.listdir(folder_path):

    # 检查是否有可用的 GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.set_default_dtype(torch.float64)  # 设置默认张量数据类型为双精度浮点数
    torch.manual_seed(33)  # 设置随机数种子为33

    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        # 读取CSV文件
        data = pd.read_csv(file_path, skiprows=0)
        # 选择第11列及后续数据
        data = data.iloc[:, 9:]  # iloc是基于0索引的
        data = data.to_numpy()


        class FullyConnectedNetwork(torch.nn.Module):
            def __init__(self):
                super(FullyConnectedNetwork, self).__init__()
                # 初始化参数
                # self.kan_r = KAN(width=[1, 1], grid=15, k=3, seed=0, base_fun=lambda x: x ** 2.2, device=device)
                # self.kan_g = KAN(width=[1, 1], grid=15, k=3, seed=0, base_fun=lambda x: x ** 2.2, device=device)
                # self.kan_b = KAN(width=[1, 1], grid=15, k=3, seed=0, base_fun=lambda x: x ** 2.2, device=device)
                self.kan_r = KAN(width=[1, 1], grid=15, k=3, seed=0, base_fun=base_function, device=device,
                                 grid_range=[-1, 2])
                self.kan_g = KAN(width=[1, 1], grid=15, k=3, seed=0, base_fun=base_function, device=device,
                                 grid_range=[-1, 2])
                self.kan_b = KAN(width=[1, 1], grid=15, k=3, seed=0, base_fun=base_function, device=device,
                                 grid_range=[-1, 2])
                self.fc1 = nn.Linear(input_features, 300)
                self.fc2 = nn.Linear(300, 300)
                self.fc3 = nn.Linear(300, output_features)
                self.activation = nn.Tanh()
                self.output_activation = nn.Sigmoid()

                # 已知的 3x3 转换矩阵
                self.conversion_matrix = nn.Parameter(torch.tensor(np.array(rgb_m), device=device),
                                                      requires_grad=False)

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
                Y_base = torch.matmul(Y_base_centered, components.T) * muti

                # 神经网络部分
                x = self.activation(self.fc1(x))
                x = self.activation(self.fc2(x))
                x = self.output_activation(self.fc3(x))

                x = Y_base + 100 * x - 50
                return x


        # 选择特定行和列,读取数据并将它们移动到指定设备
        rows_train = np.r_[0, 4, 20, 100, 124:248]  # 第1，125：248行
        x_signals = data[np.ix_(rows_train, np.r_[0:3])]
        y_lut = data[np.ix_(rows_train, np.r_[27:428])]

        rows_test = np.r_[326:526]
        rgbtst = data[np.ix_(rows_test, np.r_[0:3])]
        pcagt = data[np.ix_(rows_test, np.r_[27:428])]
        rgb_m = data[np.ix_(np.r_[20, 4, 100], np.r_[27:428])]
        cmf = np.loadtxt('ciexyz31_1s.txt')

        # 创建 PCA 实例
        pca = PCA()

        # 拟合数据
        pca.fit(y_lut)

        # 获取累计解释方差
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

        # 确定解释 99.99% 方差所需的最小主成分数目
        num_components = np.argmax(cumulative_variance >= 0.9999) + 1
        # num_components = 20
        print(f"Number of components needed to explain 99.99% of the variance: {num_components}")

        # 7. 重新创建 PCA 对象，使用选择的主成分数目
        pca = PCA(n_components=num_components)
        pca = pca.fit(y_lut)
        xyzs = pca.transform(y_lut) * muti
        xyzgt = pca.transform(pcagt) * muti

        # 将 PCA 组件和均值转换为 PyTorch 张量
        components = torch.tensor(pca.components_, dtype=torch.float64, device=device)
        mean = torch.tensor(pca.mean_, dtype=torch.float64, device=device)

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

        modelpath = os.path.join(result_path, filename)

        for modelname in os.listdir(modelpath):

            if modelname.endswith('.pt'):
                model_path = os.path.join(modelpath, modelname)
                model = FullyConnectedNetwork().to(device)
                # 读取CSV文件
                model = torch.load(model_path)

                # # 绘制trc示意图
                # plt.figure(figsize=(6, 6))
                # inputs = model.kan_r.spline_preacts[0][:, 0, 0].cpu().numpy()
                # outputs = model.kan_r.spline_postacts[0][:, 0, 0].cpu().numpy()
                # rank = np.argsort(inputs)
                # inputs = inputs[rank]
                # outputs = outputs[rank]
                # outputs -= np.min(outputs)
                # outputs /= np.max(outputs)
                # plt.plot(inputs, outputs, color='r', linewidth=10)
                # plt.axis('equal')
                # plt.xlim(-0.02, 1.02)
                # plt.ylim(-0.02, 1.02)
                # bwith = 4  # 边框宽度设置为2
                # ax = plt.gca()  # 获取边框
                # ax.spines['bottom'].set_linewidth(bwith)
                # ax.spines['left'].set_linewidth(bwith)
                # ax.spines['top'].set_linewidth(bwith)
                # ax.spines['right'].set_linewidth(bwith)
                # plt.xticks([])
                # plt.yticks([])
                # plt.savefig(f'figures\\R_trc.svg', dpi=300, format="svg")
                # plt.close()
                # plt.figure(figsize=(6, 6))
                # inputs = model.kan_g.spline_preacts[0][:, 0, 0].cpu().numpy()
                # outputs = model.kan_g.spline_postacts[0][:, 0, 0].cpu().numpy()
                # rank = np.argsort(inputs)
                # inputs = inputs[rank]
                # outputs = outputs[rank]
                # outputs -= np.min(outputs)
                # outputs /= np.max(outputs)
                # plt.plot(inputs, outputs, color='g', linewidth=10)
                # plt.axis('equal')
                # plt.xlim(-0.02, 1.02)
                # plt.ylim(-0.02, 1.02)
                # bwith = 4  # 边框宽度设置为2
                # ax = plt.gca()  # 获取边框
                # ax.spines['bottom'].set_linewidth(bwith)
                # ax.spines['left'].set_linewidth(bwith)
                # ax.spines['top'].set_linewidth(bwith)
                # ax.spines['right'].set_linewidth(bwith)
                # plt.xticks([])
                # plt.yticks([])
                # plt.savefig(f'figures\\G_trc.svg', dpi=300, format="svg")
                # plt.close()
                # plt.figure(figsize=(6, 6))
                # inputs = model.kan_b.spline_preacts[0][:, 0, 0].cpu().numpy()
                # outputs = model.kan_b.spline_postacts[0][:, 0, 0].cpu().numpy()
                # rank = np.argsort(inputs)
                # inputs = inputs[rank]
                # outputs = outputs[rank]
                # outputs -= np.min(outputs)
                # outputs /= np.max(outputs)
                # plt.plot(inputs, outputs, color='b', linewidth=10)
                # plt.axis('equal')
                # plt.xlim(-0.02, 1.02)
                # plt.ylim(-0.02, 1.02)
                # bwith = 4  # 边框宽度设置为2
                # ax = plt.gca()  # 获取边框
                # ax.spines['bottom'].set_linewidth(bwith)
                # ax.spines['left'].set_linewidth(bwith)
                # ax.spines['top'].set_linewidth(bwith)
                # ax.spines['right'].set_linewidth(bwith)
                # plt.xticks([])
                # plt.yticks([])
                # plt.savefig(f'figures\\B_trc.svg', dpi=300, format="svg")
                # plt.close()

                out = model(torch.from_numpy(x_tst).to(device))
                start_time = time.perf_counter()
                xyzpre = out.detach().cpu().numpy()

                # 计算mse
                pcapre = pca.inverse_transform(xyzpre / muti)

                xyz = 683 * np.dot(pcapre, cmf)

                end_time = time.perf_counter()

                execution_time = end_time - start_time

                t = np.append(t,execution_time)

                np.savetxt(f'{modelpath}\\{modelname}_pre.txt', pcapre)
                rmse = np.sqrt(np.mean((pcapre - pcagt) ** 2))
                print(f"{filename}RMSE: {rmse:.8f}")

np.savetxt('t.txt', t)
