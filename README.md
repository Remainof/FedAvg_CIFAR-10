# FedAvg_CIFAR-10  
基于CIFAR-10数据集实现FedAvg（Federated Averaging）  
相关论文[*Communication-Efficient Learning of Deep Networks from Decentralized Data*](https://arxiv.org/abs/1602.05629)  
## 环境
- Pytorch
- torchvision

可将项目导入pycharm，在IDE里下载对应的包
## 数据集
[*CIFAR-10*](http://www.cs.toronto.edu/~kriz/cifar.html)是一个用于识别普适物体的小型数据集。一共包含10个类别的RGB彩色图片：飞机、汽车、鸟类、猫、鹿、狗、蛙类、马、船和卡车。每个图片的尺寸为32×32，每个类别有6000个图像，数据集中一共有50000张训练图片和10000张测试图片。

## 项目结构
- **data/**: 存储CIFAR-10数据集到CIFAR10/子目录中  
- **models/**: 负责定义神经网络模型  
  model.py文件中定义了CNN模型结构  
- **client/**: 负责客户端相关的逻辑  
  client.py中实现了客户端的本地训练
- **server/**: 负责服务器端相关的逻辑   
  server.py中实现了FedAvg中的参数聚合    
- **utils/**: 负责数据加载和测试  
  data_utils.py实现了加载数据和测试全局模型
- **main.py**: 项目的主入口
