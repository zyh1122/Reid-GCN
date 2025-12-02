
#model_path = 'F:\\python_model\\HOReID\\HOReID\\results\\0429resnet50with-no-grad\\models\\gcn_27.pkl'
#F:\python_model\HOReID\HOReID\results\0430resnet50with-no-grad-0pht\models\gcn_12.pkl

import torch
from thop import profile
from core.models.model_gcn import GraphConvNet, generate_adj
import itertools  # 导入 itertools 模块
import torch.nn.functional as F
# 定义邻接矩阵
node_num = 14
linked_edges = [
    [13, 0], [13, 1], [13, 2], [13, 3], [13, 4], [13, 5], [13, 6], [13, 7], [13, 8], [13, 9], [13, 10],
    [13, 11], [13, 12],  # global
    [0, 1], [0, 2],  # head
    [1, 2], [1, 7], [2, 8], [7, 8], [1, 8], [2, 7],  # body
    [1, 3], [3, 5], [2, 4], [4, 6], [7, 9], [9, 11], [8, 10], [10, 12],  # libs
]
self_connect = 1.0
adj = generate_adj(node_num, linked_edges, self_connect)

# 加载模型定义
model = GraphConvNet(adj, in_dim=2048, hidden_dim=2048, out_dim=2048, scale=1.0)

# 加载模型参数
model_path = r'F:\python_model\HOReID\HOReID\results\0428resnet34with-no-grad\models\gcn_199.pkl'

# 检查是否有可用的 CUDA 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型时指定 map_location
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

# 将模型移到适当的设备
model = model.to(device)

# 创建一个虚拟输入张量
dummy_input = [torch.randn(16, 2048).to(device) for _ in range(14)]  # 假设有 14 个节点，每个节点的特征维度为 2048

# 修改 learn_adj 方法以确保所有张量都在同一个设备上
def learn_adj(self, inputs, adj):
    # inputs [bs, k(node_num), c]
    bs, k, c = inputs.shape

    # global_features
    global_features = inputs[:, k - 1, :].unsqueeze(1).repeat([1, k, 1])  # [bs,k,2048]
    distances = torch.abs(inputs - global_features)  # [bs, k, 2048],全局特征与各关键点之间的差异

    # bottom triangle
    distances_gap = []
    position_list = []
    for i, j in itertools.product(list(range(k)), list(range(k))):  # 14 * 14 关键点之间的关系
        if i < j and (i != k - 1 and j != k - 1) and adj[i, j] > 0:
            distances_gap.append(distances[:, i, :].unsqueeze(1) - distances[:, j, :].unsqueeze(1))  # append，做边，关键点之间的关系
            position_list.append([i, j])
    distances_gap = 15 * torch.cat(distances_gap, dim=1)  # [bs, edge_number, 2048] 每个边有2048维特征

    adj_tmp = self.sigmoid(self.scale * self.fc_direct(
        self.bn_direct(distances_gap.transpose(1, 2)).transpose(1, 2))).squeeze()  # [bs, edge_number]

    # re-assign
    adj2 = torch.ones([bs, k, k], device=device)  # 确保 adj2 在正确的设备上
    for indx, (i, j) in enumerate(position_list):
        adj2[:, i, j] = adj_tmp[:, indx] * 2
        adj2[:, j, i] = (1 - adj_tmp[:, indx]) * 2

    mask = adj.unsqueeze(0).repeat([bs, 1, 1]).to(device)  # 确保 mask 在正确的设备上
    new_adj = adj2 * mask  # 使用掩码矩阵，把没有用的边归零
    new_adj = F.normalize(new_adj, p=1, dim=2)  # [16,14,14]

    return new_adj

# 替换模型中的 learn_adj 方法
model.adgcn1.learn_adj = learn_adj.__get__(model.adgcn1)
model.adgcn2.learn_adj = learn_adj.__get__(model.adgcn2)

# 计算参数量
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Parameters: {total_params}")

# 计算 FLOPs
flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
print(f"Total FLOPs: {flops / 1e9:.2f} GFLOPs")