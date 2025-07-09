import math
import numpy as np
import scipy.sparse as sp
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import random
from sklearn.model_selection import train_test_split


class S2VGraph(object):
    def __init__(self, g, label=None, node_tags=None, node_features=None):
        '''
            g: a networkx graph（一个Networkx图对象）
            label: an integer graph label（图的整数标签）
            node_tags: a list of integer node tags （节点标签列表）
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets（节点特征张量one-hot编码）
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor（边矩阵，创建稀疏张量）
            neighbors: list of neighbors (without self-loop)（邻居列表，不含自环）
        '''
        self.label = label # 图标签
        self.g = g # 图对象
        self.node_tags = node_tags # 节点标签
        self.neighbors = [] # 邻居列表
        self.node_features = node_features # 节点特征
        self.edge_mat = 0 # 边矩阵占位符

        self.max_neighbor = 0 # 最大邻居数
        

class SparseDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, input):
        input_coal = input.coalesce() # 合并重复索引
        drop_val = F.dropout(input_coal._values(), self.p, self.training) # 对值进行dropout
        # 重建稀疏张量
        return torch.sparse.FloatTensor(input_coal._indices(), drop_val, input.shape)

# 混合类型（稀疏/密集）Dropout层
class MixedDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dense_dropout = nn.Dropout(p) # 密集数据dropout
        self.sparse_dropout = SparseDropout(p) # 稀疏数据dropout

    def forward(self, input):
        if input.is_sparse:
            return self.sparse_dropout(input)
        else:
            return self.dense_dropout(input)


# 混合类型（稀疏/密集）线性层
class MixedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features # 输入特征维度
        self.out_features = out_features # 输出特征维度
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Our fan_in is interpreted by PyTorch as fan_out (swapped dimensions) 使用Kaiming初始化权重
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            # 计算偏置的边界值
            _, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_out)
            nn.init.uniform_(self.bias, -bound, bound) # 均匀初始化偏置

    def forward(self, input):
        if self.bias is None:
            if input.is_sparse:
                # 稀疏矩阵乘法
                res = torch.sparse.mm(input, self.weight)
            else:
                # 密集矩阵乘法
                res = input.matmul(self.weight)
        else:
            if input.is_sparse:
                # 稀疏矩阵乘法加偏置
                res = torch.sparse.addmm(self.bias.expand(input.shape[0], -1), input, self.weight)
            else:
                # 密集矩阵乘法加偏置
                res = torch.addmm(self.bias, input, self.weight)
        return res

    # 额外信息表示
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
                self.in_features, self.out_features, self.bias is not None)


# 将稀疏矩阵转换为PyTorch稀疏张量
def sparse_matrix_to_torch(X):
    coo = X.tocoo() # 转换为COO格式
    indices = np.array([coo.row, coo.col]) # 获取索引
    return torch.sparse.FloatTensor( # torch.sparse
            torch.LongTensor(indices),
            torch.FloatTensor(coo.data),
            coo.shape)

# 矩阵转PyTorch张量（自动处理稀疏/密集）
def matrix_to_torch(X):
    if sp.issparse(X):
        return sparse_matrix_to_torch(X)
    else:
        return torch.FloatTensor(X)

# 通用转PyTorch张量
def to_torch(X):
    if sp.issparse(X):
        X = to_nparray(X)
    return torch.FloatTensor(X)

# 稀疏矩阵转numpy数组
def to_nparray(X):
    if sp.isspmatrix(X):
        return X.toarray()
    else: return X

# 稀疏矩阵转邻接列表
def sp2adj_lists(X):
    assert sp.isspmatrix(X), 'X should be sp.sparse' # 确保是稀疏矩阵
    adj_lists = []
    if sp.isspmatrix(X):
        for i in range(X.shape[0]):
            # 获取每个节点的邻居索引
            neighs = list( X[i,:].nonzero()[1] )
            adj_lists.append(neighs)
        return adj_lists
    else:
        return None


# 加载数据集
def load_dataset(dataset, data_dir='data'):
    from pathlib import Path
    import pickle
    import sys

    sys.path.append('data') # for pickle.load

    data_dir = Path(data_dir)
    suffix = '_25c.SG'
    graph_name = dataset + suffix
    path_to_file = data_dir / graph_name
    with open(path_to_file, 'rb') as f:
        graph = pickle.load(f)
    return graph


# 加载最新的模型检查点
def load_latest_ckpt(model_name, dataset, ckpt_dir='./checkpoints'):
    from pathlib import Path
    ckpt_dir = Path(ckpt_dir)
    ckpt_files = []
    for p in ckpt_dir.iterdir():
        if model_name in str(p) and dataset in str(p):
            ckpt_files.append(str(p)) 
    if len(ckpt_files) > 0:
        ckpt_file = sorted(ckpt_files, key=lambda x: x[-22:])[-1] 
    else: raise FileNotFoundError
    print('checkpoint file:', ckpt_file)
    import torch
    state_dict = torch.load(ckpt_file)
    return state_dict  


# 归一化稀疏矩阵（行归一化）
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

# 稀疏矩阵转PyTorch稀疏张量
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# 邻接矩阵处理（对称化+归一化）
def adj_process(adj):
    """build symmetric adjacency matrix"""
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj

# GIN数据准备函数
def gin_data_preparation(dataset, num_classes=2):
    graph = load_dataset(dataset) # 加载数据集
    
    influ_mat_list = copy.copy(graph.influ_mat_list) # 复制影响矩阵
    G = nx.from_scipy_sparse_matrix(graph.adj_matrix) # 从稀疏矩阵创建图
    degrees = np.array([val for (node, val) in G.degree()]) # 计算节点度

    # 提取种子向量和影响向量
    seed_vec = influ_mat_list[:, :, 0] # 初始状态
    influ_vec = influ_mat_list[:, :, -1] # 最终状态
    
    seed_vec = [torch.FloatTensor(i) for i in seed_vec] # 转张量列表
    influ_vec = [torch.FloatTensor(i) for i in influ_vec]
    g_list = []
    # 为每个种子向量创建图对象
    for x in seed_vec:
        # 创建节点特征（one-hot编码）
        temp_feature = F.one_hot(x.to(torch.long), num_classes).to(torch.float)
        g_list.append(S2VGraph(G, 0, node_tags=None, node_features=temp_feature))
    
    #add labels and edge_mat 为图对象添加标签和边矩阵
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))] # 初始化邻居列表
        # 添加边关系
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list) # 记录最大邻居数

        # 创建边矩阵（包含双向边）
        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        #deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0,1) # 转置为[2, E]形状

    # 分割训练集和测试集
    train_g_list, test_g_list, train_x, test_x, train_y, test_y = train_test_split(g_list, seed_vec, influ_vec, test_size=0.13)
    
    return train_g_list, test_g_list, train_x, test_x, train_y, test_y


# 逆问题数据集类
class InverseProblemDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.graph = load_dataset(dataset) # 加载图数据
        
        self.data = self.cache(self.graph) # 缓存数据
        
    def cache(self, graph):
        # 只取前50个影响矩阵
        graph.influ_mat_list = graph.influ_mat_list[:50]
        influ_mat_list = copy.copy(graph.influ_mat_list)

        # 提取种子向量和影响向量
        seed_vec = influ_mat_list[:, :, 0]
        influ_vec = influ_mat_list[:, :, -1]
        # 组合成向量对
        vec_pairs = torch.Tensor(np.stack((seed_vec, influ_vec), -1))
        return vec_pairs
        
    def __getitem__(self, item):
        vec_pair = self.data[item] # 获取单个样本
        return vec_pair
    
    def __len__(self):
        # 数据集大小
        return len(self.data)

# 扩散模型评估函数
def diffusion_evaluation(adj_matrix, seed, diffusion='LT'):
    """
    评估扩散模型效果
    Args:
        adj_matrix: 邻接矩阵
        seed: 种子节点集合
        diffusion: 扩散模型类型（'LT', 'IC', 'SIS'）
    Returns:
        平均感染节点数
    """
    total_infect = 0
    G = nx.from_scipy_sparse_matrix(adj_matrix) # 创建图

    # 运行10次取平均
    for i in range(10):
        if diffusion == 'LT': # 线性阈值模型
            model = ep.ThresholdModel(G)
            config = mc.Configuration()
            for n in G.nodes():
                config.add_node_configuration("threshold", n, 0.5)
        elif diffusion == 'IC': # 独立级联模型
            model = ep.IndependentCascadesModel(G)
            config = mc.Configuration()
            for e in G.edges():
                config.add_edge_configuration("threshold", e, 1/nx.degree(G)[e[1]])
        elif diffusion == 'SIS': # SIS流行病模型
            model = ep.SISModel(G)
            config = mc.Configuration()
            config.add_model_parameter('beta', 0.001)
            config.add_model_parameter('lambda', 0.001)
        else:
            raise ValueError('Only IC, LT and SIS are supported.')

        config.add_model_initial_configuration("Infected", seed)

        model.set_initial_status(config)

        iterations = model.iteration_bunch(100)

        node_status = iterations[0]['status']

        seed_vec = np.array(list(node_status.values()))

        for j in range(1, len(iterations)):
            node_status.update(iterations[j]['status'])


        inf_vec = np.array(list(node_status.values()))
        inf_vec[inf_vec == 2] = 1

        total_infect += inf_vec.sum()
    
    return total_infect/10