import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout # dropout率

        # 创建多头注意力层
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        # 将每个注意力层注册为子模块
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # 输出层注意力
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        # 输入dropout
        x = F.dropout(x, self.dropout, training=self.training)
        # 多头注意力：拼接所有头的输出
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        # 再次dropout
        x = F.dropout(x, self.dropout, training=self.training)
        # 输出层注意力 + ELU激活
        x = F.elu(self.out_att(x, adj))
        # Sigmoid激活得到最终输出
        return torch.sigmoid(x)


class SpGAT(nn.Module):
    """
    稀疏版本的图注意力网络(GAT)
    用于处理大规模稀疏图

    参数同GAT
    """
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout
        # 第一层多头注意力
        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        # 第二层多头注意力
        self.attentions1 = [SpGraphAttentionLayer(nhid * nheads, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        # 注册第一层注意力模块
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        # 注册第二层注意力模块
        for i, attention in enumerate(self.attentions1):
            self.add_module('attention1_{}'.format(i), attention)
        # 输出层注意力
        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)
        

    def forward(self, x, adj):
        # 输入dropout
        x = F.dropout(x, self.dropout, training=self.training)
        # 第一层多头注意力 + ELU激活
        x = F.elu(torch.cat([att(x, adj) for att in self.attentions], dim=1))
        # 再次dropout
        x = F.dropout(x, self.dropout, training=self.training)
        # 输出层注意力 + ELU激活
        x = F.elu(self.out_att(x, adj))
        return x


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    密集图注意力层

    参数:
    in_features: 输入特征维度
    out_features: 输出特征维度
    dropout: Dropout率
    alpha: LeakyReLU的负斜率
    concat: 是否在输出时使用ELU激活
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # 特征变换矩阵W
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        # 注意力机制参数向量a
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # LeakyReLU激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # 特征变换: h * W
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        # 计算注意力分数
        e = self._prepare_attentional_mechanism_input(Wh)

        # 创建掩码: 有边连接的位置为0，无边连接的位置为负无穷
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec) # 应用邻接矩阵掩码
        # 计算softmax归一化的注意力权重
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        # 应用注意力权重: 聚合邻居信息
        h_prime = torch.matmul(attention, Wh)

        # 输出激活
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
    
    
class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b) # 保存必要张量用于反向传播
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors # 获取保存的必要张量
        grad_values = grad_b = None # 初始化梯度张量
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :] # 将二维坐标装换位一维索引
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

    
class SpGraphAttentionLayer(nn.Module):
    """
    稀疏图注意力层（Sparse Graph Attention Layer），基于论文《Graph Attention Networks》(GAT) 的稀疏矩阵优化实现。
    该层通过稀疏矩阵运算实现高效的注意力机制，适用于大规模图结构数据，支持归纳学习（Inductive Learning）和直推式学习（Transductive Learning）。
    核心思想：通过自注意力机制动态计算邻接节点的重要性权重，无需预先定义图结构或矩阵分解。

    参数:
        in_features (int): 输入特征维度
        out_features (int): 输出特征维度（单头注意力）
        dropout (float): 注意力系数的Dropout比例
        alpha (float): LeakyReLU的负斜率参数
        concat (bool): 若为True，使用ELU激活函数（中间层）；若为False，不激活（最后一层）
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0] # 节点数量
        if adj.layout == torch.sparse_coo:
            edge = adj.indices() # 稀疏矩阵直接获取索引
        else:
            edge = adj.nonzero().t() # 稠密矩阵转换为边索引

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        # 计算归一化因子
        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        # 消息传递与聚合
        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'