# 导入本地训练函数、模型聚合函数、模型定义以及数据加载和测试工具
import random
import torch
from client.client import local_train
from server.server import aggregate
from models.model import Net
from utils.data_utils import load_data, test_model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 加载训练集和测试集数据
    trainloader, testloader = load_data()


    num_clients = 20            # 设置客户端数量
    num_selected_clients=5      #每轮训练选择的客户端数量
    rounds=10                   #训练轮次
    epochs=2                    #每轮训练的迭代次数为为2

    # 初始化全局模型
    global_model = Net().to(device)
    for round in range(rounds):
        selected_clients = random.sample(range(num_clients), num_selected_clients)
        client_models = []  #用来存储客户端每一轮训练完后的参数

        print(f"Round {round + 1}: Selected clients {selected_clients}")
        # 模拟每个客户端的本地训练
        for client in range(num_clients):
            # 创建客户端模型并加载全局模型的参数
            local_model = Net().to(device)
            local_model.load_state_dict(global_model.state_dict())

            # 客户端进行本地训练，并返回训练后的模型参数
            client_state_dict = local_train(local_model, trainloader,epochs,device)
            client_models.append(client_state_dict)

        # 服务器端聚合所有客户端的模型参数，更新全局模型
        global_model = aggregate(global_model, client_models)

        print(f"Round {round + 1} complete")  # 打印当前轮次完成的信息

    # 训练完成后，用测试集测试最终的全局模型
    test_model(global_model, testloader,device)

if __name__ == "__main__":
    main()  # 启动主程序
