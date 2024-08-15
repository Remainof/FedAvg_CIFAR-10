import torch.optim as optim
import torch.nn as nn

# 定义客户端的本地训练函数
def local_train(client_model, trainloader, epochs, device):
    # 定义损失函数为交叉熵损失
    criterion = nn.CrossEntropyLoss()

    # 定义优化器为随机梯度下降（SGD），学习率为0.001，动量为0.9
    optimizer = optim.SGD(client_model.parameters(), lr=0.001, momentum=0.9)

    client_model.train()  # 将模型设置为训练模式
    for epoch in range(epochs):  # 对于指定的训练轮数进行循环
        running_loss = 0.0  # 初始化累计损失
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data  # 获取输入数据和标签
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # 梯度清零
            outputs = client_model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数
            running_loss += loss.item()  # 累计损失
    return client_model.state_dict()  # 返回训练后的模型参数
