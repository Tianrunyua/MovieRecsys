import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from sysMovie.MFMGNN_IPTVRS.Basicgcn import Base_gcn
from sysMovie.MFMGNN_IPTVRS.ItemGCN import ItemGCN
from sysMovie.MFMGNN_IPTVRS.ItemGateGAT import ItemGateGAT
from sysMovie.MFMGNN_IPTVRS.ItemNormGAT import ItemNormGAT
import matplotlib
matplotlib.use('Agg')


class GCN(torch.nn.Module):
    def __init__(self,datasets, batch_size, num_user, num_item, dim_id, aggr_mode, num_layer, has_id,
                 dim_latent,device = None,features_dim=None):
        super(GCN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.datasets = datasets
        self.dim_id = dim_id
        if self.datasets =='Tiktok':
             self.dim_feat = 128
        elif self.datasets == 'Movielens' or self.datasets == 'Alishop' or self.datasets=='Test' or self.datasets=='IPTV':
             self.dim_feat = features_dim
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode
        self.num_layer = num_layer
        self.has_id = has_id
        self.device = device

        self.preference = nn.Parameter(
                nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user, self.dim_latent),dtype=torch.float32, requires_grad=True), gain=1).to(self.device))
        self.MLP = nn.Linear(self.dim_feat, 4*self.dim_latent)
        self.MLP_1 = nn.Linear(4*self.dim_latent, self.dim_latent)
        self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)

    def forward(self, edge_index,features):
        temp_features = self.MLP_1(F.leaky_relu(self.MLP(features))) if self.dim_latent else features
        # temp_features = F.normalize(temp_features)
        x = torch.cat((self.preference, temp_features), dim=0).to(self.device)
        x = F.normalize(x).to(self.device)
        h = self.conv_embed_1(x, edge_index)  # equation 1
        h_1 = self.conv_embed_1(h, edge_index)
        x_hat = h + x + h_1
        return x_hat, self.preference


class item_GCN(torch.nn.Module):
    def __init__(self,datasets,aggr_mode, dim_latent,device = None,features_dim=None,att='gategat'):
        super(item_GCN, self).__init__()
        self.datasets = datasets
        if self.datasets =='Tiktok':
             self.dim_feat = 128
        elif self.datasets == 'Movielens' or self.datasets == 'Alishop' or self.datasets=='Test' or self.datasets=='IPTV':
             self.dim_feat = features_dim
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode
        self.device = device
        self.att = att
        if self.dim_latent:
            self.MLP = nn.Linear(self.dim_feat, 4*self.dim_latent)
            self.MLP_1 = nn.Linear(4*self.dim_latent, self.dim_latent)
            if att == 'normgat':
                self.conv_embed_1 = ItemNormGAT(self.dim_latent, self.dim_latent,gat='concat', aggr=self.aggr_mode)
            elif att == 'gategat':
                self.conv_embed_1 = ItemGateGAT(self.dim_latent, self.dim_latent, gate = 'concat',aggr=self.aggr_mode)
            else:
                self.conv_embed_1 = ItemGCN(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)

        else:
            if att == 'normgat':
                self.conv_embed_1 = ItemNormGAT(self.dim_feat, self.dim_feat,gat='concat', aggr=self.aggr_mode)
            elif att == 'gategat':
                self.conv_embed_1 = ItemGateGAT(self.dim_feat, self.dim_feat,gate = 'concat', aggr=self.aggr_mode)
            else:
                self.conv_embed_1 = ItemGCN(self.dim_feat, self.dim_feat, aggr=self.aggr_mode)

    def forward(self, edge_index,features):
        temp_features = self.MLP_1(F.leaky_relu(self.MLP(features))) if self.dim_latent else features
        x = temp_features
        h = self.conv_embed_1(x, edge_index) # equation 1
        return h


class User_Graph_sample(torch.nn.Module):
        def __init__(self, num_user, aggr_mode, dim_latent):
            super(User_Graph_sample, self).__init__()
            self.num_user = num_user
            self.dim_latent = dim_latent
            self.aggr_mode = aggr_mode

        def forward(self, features, user_graph, user_matrix):
            index = user_graph
            u_features = features[index]
            user_matrix = user_matrix.unsqueeze(1)
            # pdb.set_trace()
            u_pre = torch.matmul(user_matrix, u_features)
            u_pre = u_pre.squeeze()
            return u_pre


class MultiGraph(torch.nn.Module):
    def __init__(self, features, edge_index,batch_size, num_user, num_item, aggr_mode, construction,
                 num_CFGCN_layer, num_itemgraph_layer, has_id, dim_latent, reg_weight,
                 user_item_dict,dataset,item_graph_edge_index_com,item_graph_edge_index_v,
                 item_graph_edge_index_t,user_topk,num_pre,user_graph,device):
        super(MultiGraph, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.device =device
        self.aggr_mode = aggr_mode
        self.num_layer = num_CFGCN_layer
        self.num_itemgraph_layer = num_itemgraph_layer
        self.dataset = dataset
        self.construction = construction
        self.reg_weight = reg_weight
        self.user_item_dict = user_item_dict
        self.v_preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(np.random.randn(self.num_user,128), dtype=torch.float32, requires_grad=True)))
        self.t_preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(np.random.randn(self.num_user, 128), dtype=torch.float32, requires_grad=True)))
        self.v_rep = None#(所有项目+用户的表示)
        self.t_rep = None
        self.device = device
        self.dim_latent = dim_latent
        self.dim_feat=dim_latent
        v_feat,t_feat = features
        self.num_pre=num_pre
        # 得到最后的用户-项目二部图
        self.edge_index = torch.tensor(edge_index,dtype=torch.long).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)
        self.user_graph = user_graph
        # 得到项目知识图
        self.item_graph_edge_index_com = torch.tensor(item_graph_edge_index_com, dtype=torch.long).t().contiguous().to(self.device)
        self.item_graph_edge_index_v_sim = torch.tensor(item_graph_edge_index_v, dtype=torch.long).t().contiguous().to(self.device)
        self.item_graph_edge_index_t_sim = torch.tensor(item_graph_edge_index_t, dtype=torch.long).t().contiguous().to(self.device)
       # 构建用户对不同模态的偏好信息
        self.weight_u = nn.Parameter(nn.init.xavier_normal_(torch.tensor(np.random.randn(self.num_user, 2, 1), dtype=torch.float32, requires_grad=True)))
        self.weight_u.data = F.softmax(self.weight_u.data, dim=1)
        self.user_weight_matrix = nn.Parameter(nn.init.xavier_normal_(torch.tensor(np.random.randn(self.num_user, user_topk), dtype=torch.float32, requires_grad=True)))
        # self.user_weight_matrix.data = F.softmax(self.user_weight_matrix.data, dim=1)
        # 构建项目对两个不同项目关系图的项目表示的加权求和系数
        self.weight_vgraph = nn.Parameter(nn.init.xavier_normal_(torch.tensor(np.random.randn(self.num_item, 2, 1),dtype=torch.float32, requires_grad=True)))
        self.weight_vgraph.data = F.softmax(self.weight_vgraph.data, dim=1)
        self.weight_tgraph = nn.Parameter(nn.init.xavier_normal_(torch.tensor(np.random.randn(self.num_item, 2, 1), dtype=torch.float32, requires_grad=True)))
        self.weight_tgraph.data = F.softmax(self.weight_tgraph.data, dim=1)
        # 读取三个模态的特征
        self.MLP_user = nn.Linear(self.dim_latent * 2, self.dim_latent)
        self.v_feat = torch.tensor(v_feat, dtype=torch.float).to(self.device)
        self.dim_v = self.v_feat.shape[1]
        self.t_feat = torch.tensor(t_feat, dtype=torch.float).to(self.device)
        self.dim_t = self.t_feat.shape[1]
        self.dim_cat=self.dim_t+self.dim_v
        self.tra_v=nn.Linear(self.dim_latent,self.dim_v)
        self.tra_t = nn.Linear(self.dim_latent, self.dim_t)

        # 构建模态对应的图卷积网络
        self.MLP_user = nn.Linear(self.dim_latent*5, self.dim_latent)
        self.v_gcn = GCN(self.dataset,batch_size, num_user, num_item, dim_latent, self.aggr_mode,
                               num_layer=self.num_layer, has_id=has_id, dim_latent=self.dim_latent,
                               device=self.device, features_dim=self.dim_v)  # 256)
        self.t_gcn = GCN(self.dataset,batch_size, num_user, num_item, dim_latent, self.aggr_mode,
                               num_layer=self.num_layer, has_id=has_id, dim_latent=self.dim_latent,
                               device=self.device, features_dim=self.dim_t)
        self.v_gcn_hop = item_GCN(self.dataset, self.aggr_mode, dim_latent=None,device=self.device, features_dim=self.dim_v)  # 256)
        self.t_gcn_hop = item_GCN(self.dataset, self.aggr_mode,dim_latent=None,device=self.device, features_dim=self.dim_t)
        self.user_graph = User_Graph_sample(num_user, 'add', self.dim_latent)
        # self.result_embed = nn.Parameter(nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user + num_item, dim_latent)))).to(self.device)

    def computer_itemfeature(self):
        self.v_feat_sim = self.v_gcn_hop(self.item_graph_edge_index_v_sim, self.v_feat)
        self.v_feat_com = self.v_gcn_hop(self.item_graph_edge_index_com, self.v_feat)
        self.t_feat_sim = self.t_gcn_hop(self.item_graph_edge_index_t_sim, self.t_feat)
        self.t_feat_com = self.t_gcn_hop(self.item_graph_edge_index_com, self.t_feat)
        self.v_feat_hero = torch.matmul(torch.cat((torch.unsqueeze(self.v_feat_sim, 2),
                                                   torch.unsqueeze(self.v_feat_com, 2))
                                                  , dim=2), self.weight_vgraph)
        self.t_feat_hero = torch.matmul(torch.cat((torch.unsqueeze(self.t_feat_sim, 2),
                                                   torch.unsqueeze(self.t_feat_com, 2))
                                                  , dim=2), self.weight_tgraph)
        self.t_feat_hero = torch.squeeze(self.t_feat_hero)
        self.v_feat_hero = torch.squeeze(self.v_feat_hero)
        return self.v_feat_hero,self.t_feat_hero

    def computer_itematt(self,edge_index,v_feat_hero,t_feat_hero):
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        edge_index = torch.cat((edge_index, edge_index[[1, 0]]), dim=1)
        v_feat = self.v_feat + v_feat_hero
        t_feat = self.t_feat + t_feat_hero
        self.v_rep, self.v_preference = self.v_gcn(edge_index, v_feat)
        self.t_rep, self.t_preference = self.t_gcn(edge_index, t_feat)
        # ########################################### multi-modal information construction
        if self.construction == 'weighted_sum':
            self.v_rep = torch.unsqueeze(self.v_rep, 2)
            self.t_rep = torch.unsqueeze(self.t_rep, 2)
            self.user_rep = torch.matmul(torch.cat((self.v_rep[:self.num_user], self.t_rep[:self.num_user])
                                                   , dim=2), self.weight_u)
            self.user_rep = torch.squeeze(self.user_rep)
        return self.user_rep,self.v_rep,self.t_rep

    def forward(self,curuser,edge_index,user_graph,v_feat_hero,t_feat_hero):
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        edge_index = torch.cat((edge_index, edge_index[[1, 0]]), dim=1)
        v_feat = self.v_feat + v_feat_hero
        t_feat = self.t_feat + t_feat_hero
        self.v_rep, self.v_preference = self.v_gcn(edge_index, v_feat)
        self.t_rep, self.t_preference = self.t_gcn(edge_index, t_feat)
        # ########################################### multi-modal information construction
        representation = self.v_rep + self.t_rep
        if self.construction == 'weighted_sum':
            self.v_rep = torch.unsqueeze(self.v_rep, 2)
            self.t_rep = torch.unsqueeze(self.t_rep, 2)
            self.user_rep = torch.matmul(torch.cat((self.v_rep[:self.num_user], self.t_rep[:self.num_user])
                                              ,dim=2),self.weight_u)
            self.user_rep = torch.squeeze(self.user_rep)
        item_rep = representation[self.num_user:]
        index = torch.tensor(user_graph)[curuser].tolist()
        u_features = self.user_rep[index]
        # user_matrix = self.user_weight_matrix[curuser].unsqueeze(1)
        user_matrix = self.user_weight_matrix[curuser]
        u_pre = torch.matmul(user_matrix, u_features)
        self.user_rep = self.user_rep[curuser] + u_pre.squeeze()
        # self.result_embed = torch.cat((self.user_rep, item_rep), dim=0)
        # h_u1 = self.user_graph(self.user_rep, self.user_graph,curuser, self.user_weight_matrix.to(self.device))
        # self.user_rep = self.user_rep + h_u1
        # self.result_embed = torch.cat((self.user_rep, item_rep), dim=0)
        score_matrix = torch.matmul(self.user_rep, item_rep.t())
        col = self.user_item_dict[curuser]
        col = torch.LongTensor(list(col)) - self.num_user
        score_matrix[col] = 1e-5
        score_of_rank_list , index_of_rank_list = torch.topk(score_matrix, 6259)
        return index_of_rank_list,score_of_rank_list


