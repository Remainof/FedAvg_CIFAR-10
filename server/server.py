import torch

# 定义服务器端的模型聚合函数
def aggregate(global_model, client_state_dicts):
    # 获取全局模型的状态字典（参数）
    global_dict = global_model.state_dict()

    # 遍历每一个参数
    for k in global_dict.keys():
        # 将所有客户端的相同参数进行平均，更新全局模型的参数
        global_dict[k] = torch.stack([client_state_dicts[i][k].float() for i in range(len(client_state_dicts))], 0).mean(0)

    # 将更新后的参数加载回全局模型
    global_model.load_state_dict(global_dict)

    return global_model  # 返回更新后的全局模型
