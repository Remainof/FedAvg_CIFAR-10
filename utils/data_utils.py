import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset


# 定义数据加载函数
def load_data(num_clients,test):
    # 数据预处理：将图片转换为张量并归一化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if (test==True):
        testset = torchvision.datasets.CIFAR10(root='.\\data\\CIFAR10', train=False,download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100,shuffle=True, num_workers=2)
        return testloader
    else:
        trainset = torchvision.datasets.CIFAR10(root='.\\data\\CIFAR10',train=False,download=True, transform=transform)
        # 将数据集随机划分为 num_clients 份
        data_per_client = len(trainset)//num_clients
        indices = torch.randperm(len(trainset))   #对长度为 len(trainset) 的序列进行随机排列，返回包含这些随机排列索引的张量
        client_datasets = []
        for i in range(num_clients):
            #从 indices 中选出的部分索引来确定一个客户端（或数据子集）应当获取的样本。
            client_indices = indices[i * data_per_client:(i + 1) * data_per_client]
            client_subset = Subset(trainset, client_indices)
            client_dataloader = torch.utils.data.DataLoader(client_subset, batch_size=100, shuffle=True, num_workers=2)
            client_datasets.append(client_dataloader)

        return client_datasets


# 定义测试全局模型的函数
def test_model(global_model, testloader, device):
    correct = 0  # 初始化正确预测的计数器
    total = 0  # 初始化总计数器
    global_model.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 在评估时不计算梯度
        for data in testloader:
            images, labels = data  # 获取输入数据和标签
            images, labels = images.to(device), labels.to(device)
            outputs = global_model(images)  # 前向传播
            _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
            total += labels.size(0)  # 累计测试样本数
            correct += (predicted == labels).sum().item()  # 累计正确预测数

    # 打印模型在测试集上的准确率
    print('Accuracy of the network on the 10000 test images: %d %%' % ( 100 * correct / total))
