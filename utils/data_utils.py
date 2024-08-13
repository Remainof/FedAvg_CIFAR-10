import torch
import torchvision
import torchvision.transforms as transforms


# 定义数据加载函数
def load_data():
    # 数据预处理：将图片转换为张量并归一化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载CIFAR-10训练集
    trainset = torchvision.datasets.CIFAR10(root='.\\data\\CIFAR10',
                                            train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                              shuffle=True, num_workers=2)

    # 加载CIFAR-10测试集
    testset = torchvision.datasets.CIFAR10(root='.\\data\\CIFAR10',
                                           train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=2)

    return trainloader, testloader  # 返回训练集和测试集的数据加载器


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
