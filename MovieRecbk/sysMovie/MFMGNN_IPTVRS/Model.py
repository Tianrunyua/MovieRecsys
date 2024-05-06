import math
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dropout_adj
from torch_geometric.utils import scatter_
from torch.nn import Parameter
from Basicgcn import Base_gcn
from ItemGCN import ItemGCN
from ItemGateGAT import ItemGateGAT
from ItemNormGAT import ItemNormGAT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
import random
import os
import torch.nn as nn
from parse import parse_args
args = parse_args()
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
    def __init__(self,datasets,aggr_mode, dim_latent,device = None,features_dim=None,att='gcn'):
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
                 user_item_dict,dataset,item_graph_edge_index_com,item_graph_edge_index_v,item_graph_edge_index_t,user_topk,num_pre,device):
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
        self.v_rep,self.v_preference = None,None#(所有项目+用户的表示)
        self.a_rep,self.a_preference= None,None
        self.t_rep,self.t_preference = None,None
        self.device = device
        self.dim_latent = dim_latent
        self.dim_feat=dim_latent
        v_feat,t_feat = features
        self.num_pre=num_pre
        #得到最后的用户-项目二部图
        self.edge_index = torch.tensor(edge_index,dtype=torch.long).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)
        #得到项目知识图
        self.item_graph_edge_index_com = torch.tensor(item_graph_edge_index_com, dtype=torch.long).t().contiguous().to(self.device)
        self.item_graph_edge_index_v_sim = torch.tensor(item_graph_edge_index_v, dtype=torch.long).t().contiguous().to(self.device)
        if self.dataset!='Tiktok':
            self.item_graph_edge_index_t_sim = torch.tensor(item_graph_edge_index_t, dtype=torch.long).t().contiguous().to(self.device)

       #构建用户对不同模态的偏好信息
        self.weight_u = nn.Parameter(nn.init.xavier_normal_(torch.tensor(np.random.randn(self.num_user, 2, 1), dtype=torch.float32, requires_grad=True)))
        self.weight_u.data = F.softmax(self.weight_u.data, dim=1)
        self.user_weight_matrix = nn.Parameter(nn.init.xavier_normal_(torch.tensor(np.random.randn(self.num_user, user_topk), dtype=torch.float32, requires_grad=True)))
        # self.user_weight_matrix.data = F.softmax(self.user_weight_matrix.data, dim=1)
        #构建项目对两个不同项目关系图的项目表示的加权求和系数
        self.weight_vgraph = nn.Parameter(nn.init.xavier_normal_(torch.tensor(np.random.randn(self.num_item, 2, 1),dtype=torch.float32, requires_grad=True)))
        self.weight_vgraph.data = F.softmax(self.weight_vgraph.data, dim=1)
        self.weight_tgraph = nn.Parameter(nn.init.xavier_normal_(torch.tensor(np.random.randn(self.num_item, 2, 1), dtype=torch.float32, requires_grad=True)))
        self.weight_tgraph.data = F.softmax(self.weight_tgraph.data, dim=1)
        #读取三个模态的特征
        if self.dataset=='Tiktok'  :
            self.word_tensor = torch.tensor(t_feat).long().to(self.device)
            self.v_feat = torch.tensor(v_feat).float().to(self.device)
            self.dim_v = self.dim_t=128
            self.word_embedding = nn.Embedding(torch.max(self.word_tensor[1]) + 1, 128).to(self.device)
            nn.Parameter(nn.init.xavier_normal_(self.word_embedding.weight))
        elif self.dataset == 'Movielens' or self.dataset == 'Alishop'or self.dataset=='Test' or self.dataset=='IPTV':
            self.MLP_user = nn.Linear(self.dim_latent * 2, self.dim_latent)
            self.v_feat = torch.tensor(v_feat, dtype=torch.float).to(self.device)
            self.dim_v = self.v_feat.shape[1]
            self.t_feat = torch.tensor(t_feat, dtype=torch.float).to(self.device)
            self.dim_t = self.t_feat.shape[1]
        self.dim_cat=self.dim_t+self.dim_v
        self.tra_v=nn.Linear(self.dim_latent,self.dim_v)
        self.tra_t = nn.Linear(self.dim_latent, self.dim_t)

        #构建模态对应的图卷积网络
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
        # self.itemID_embed_v = nn.Parameter(nn.init.xavier_normal_(torch.tensor(np.random.randn(num_item, dim_latent),dtype=torch.float))).to(self.device)
        # self.itemID_embed_t = nn.Parameter(nn.init.xavier_normal_(torch.tensor(np.random.randn(num_item, dim_latent),dtype=torch.float))).to(self.device)
    def forward(self, user_nodes, pos_item_nodes, neg_item_nodes,user_graph,user_weight_matrix):
        if self.dataset == 'Tiktok':
            self.t_feat = scatter_('mean', self.word_embedding(self.word_tensor[1]), self.word_tensor[0]).to(
                self.device)
            self.item_graph_edge_index_t_sim = torch.tensor(self.get_edge_sim(self.t_feat, topk_sim=5,simval=0.85),
                                                            dtype=torch.long).t().contiguous().to(self.device)

        self.v_feat_sim = self.v_gcn_hop(self.item_graph_edge_index_v_sim,self.v_feat)
        self.v_feat_com = self.v_gcn_hop(self.item_graph_edge_index_com,self.v_feat)
        self.t_feat_sim = self.t_gcn_hop(self.item_graph_edge_index_t_sim,self.t_feat)
        self.t_feat_com = self.t_gcn_hop(self.item_graph_edge_index_com,self.t_feat)
        self.v_feat_hero = torch.matmul(torch.cat((torch.unsqueeze(self.v_feat_sim,2),
                                                  torch.unsqueeze(self.v_feat_com,2))
                                                    ,dim=2),self.weight_vgraph)
        self.t_feat_hero = torch.matmul(torch.cat((torch.unsqueeze(self.t_feat_sim, 2),
                                                  torch.unsqueeze(self.t_feat_com, 2))
                                                 , dim=2), self.weight_tgraph)
        self.t_feat_hero = torch.squeeze(self.t_feat_hero)
        self.v_feat_hero = torch.squeeze(self.v_feat_hero)
        v_feat = self.v_feat+self.v_feat_hero
        t_feat = self.t_feat+self.t_feat_hero
        self.v_rep,self.v_preference = self.v_gcn(self.edge_index,v_feat)
        self.t_rep, self.t_preference = self.t_gcn(self.edge_index,t_feat)
        # ########################################### multi-modal information construction
        representation = self.v_rep + self.t_rep
        if self.construction == 'weighted_sum':
            self.v_rep = torch.unsqueeze(self.v_rep, 2)
            self.t_rep = torch.unsqueeze(self.t_rep, 2)
            user_rep = torch.matmul(torch.cat((self.v_rep[:self.num_user], self.t_rep[:self.num_user])
                                              ,dim=2),self.weight_u)
            user_rep = torch.squeeze(user_rep)
        item_rep = representation[self.num_user:]
        h_u1 = self.user_graph(user_rep, user_graph, self.user_weight_matrix.to(self.device))
        user_rep = user_rep + h_u1
        self.result_embed = torch.cat((user_rep, item_rep), dim=0)
        #index = torch.tensor(user_graph)[user_nodes].tolist()
        #u_features = user_rep[index]
        #user_matrix = user_weight_matrix[user_nodes].unsqueeze(1)
        #u_pre = torch.matmul(user_matrix, u_features)
        #user_rep = user_rep[user_nodes] + u_pre.squeeze()
        #self.result_embed = torch.cat((user_rep, item_rep), dim=0)
        user_tensor = self.result_embed[user_nodes]
        pos_item_tensor = self.result_embed[pos_item_nodes]
        neg_item_tensor = self.result_embed[neg_item_nodes]
        pos_scores = torch.sum(user_tensor * pos_item_tensor, dim=1)
        neg_scores = torch.sum(user_tensor * neg_item_tensor, dim=1)
        return pos_scores, neg_scores

    def loss(self, data , user_graph,user_weight_matrix):
        user, pos_items, neg_items = data
        pos_scores, neg_scores = self.forward(user.to(self.device), pos_items.to(self.device),
                                              neg_items.to(self.device) ,user_graph,user_weight_matrix.to(self.device))
        loss_value = -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores)))

        reg_embedding_loss_v = (self.v_preference[user.to(self.device)] ** 2).mean()
        reg_embedding_loss_t = (self.t_preference[user.to(self.device)] ** 2).mean()
        reg_loss = self.reg_weight * (reg_embedding_loss_v + reg_embedding_loss_t+(self.weight_vgraph ** 2).mean()+(self.weight_tgraph ** 2).mean())
        if self.construction == 'weighted_sum':
            reg_loss+=self.reg_weight*(self.weight_u ** 2).mean()
            reg_loss+=self.reg_weight*(self.user_weight_matrix ** 2).mean()
            #reg_loss += self.reg_weight * (self.weight_vgraph ** 2).mean()
            #reg_loss += self.reg_weight * (self.weight_tgraph ** 2).mean()
        elif self.construction == 'cat_mlp':
            reg_loss+=self.reg_weight*(self.MLP_user.weight ** 2).mean()
        return loss_value + reg_loss, reg_loss

    def get_edge_sim(self,feat, topk_sim, simval=0.7):
        item_edge_list_feat = []
        feat = feat.cpu()
        feat_norm = torch.tensor(feat, dtype=torch.float) \
            .div(torch.norm(torch.tensor(feat, dtype=torch.float), p=2, dim=-1, keepdim=True))
        feat_norm[torch.isnan(feat_norm)] = 0.
        sim = torch.mm(feat_norm, feat_norm.transpose(1, 0))
        sim_diag = torch.diag_embed(torch.diag(sim))
        sim = sim - sim_diag
        knn_val, knn_ind = torch.topk(sim, topk_sim, dim=-1)
        for itemid, (neighbour_items, neighbour_vals) in enumerate(zip(knn_ind, knn_val)):
            if (max(neighbour_vals) < simval):
                continue
            else:
                for neighbour_item, neighbour_val in zip(neighbour_items, neighbour_vals):
                    if (neighbour_val < simval):
                        break
                    else:
                        item_edge_list_feat.append([itemid, neighbour_item])
        return np.array(item_edge_list_feat)
    def gene_ranklist(self, val_data, test_data, step=200, topk=50):
        user_tensor = self.result_embed[:self.num_user].cpu()
        item_tensor = self.result_embed[self.num_user:self.num_user + self.num_item].cpu()
        start_index = 0
        end_index = self.num_user if step == None else step

        all_index_of_rank_list_tra = torch.LongTensor([])
        all_index_of_rank_list_vt = torch.LongTensor([])
        all_index_of_rank_list_tt = torch.LongTensor([])
        while end_index <= self.num_user and start_index < end_index:
            temp_user_tensor = user_tensor[start_index:end_index]
            # 计算当前批次用户与所有项目的相似性得分
            score_matrix_tra = torch.matmul(temp_user_tensor, item_tensor.t())
            score_matrix_vt = score_matrix_tra.clone().detach()
            score_matrix_tt = score_matrix_tra.clone().detach()

            # 得到当前批次的用户的topk特能交互的项目
            _, index_of_rank_list_tra = torch.topk(score_matrix_tra, topk)
            all_index_of_rank_list_tra = torch.cat(
                (all_index_of_rank_list_tra, index_of_rank_list_tra.cpu() + self.num_user),
                dim=0)
            # 将验证集和测试集的[用户数，项目数]的得分矩阵中训练集交互过的项目的得分设置为0，以防训练集的项目造成干扰
            # user_item_dict_train
            for row, col in self.user_item_dict.items():
                if row >= start_index and row < end_index:
                    row -= start_index
                    col = torch.LongTensor(list(col)) - self.num_user
                    score_matrix_vt[row][col] = 1e-5
                    score_matrix_tt[row][col] = 1e-5
            # 将测试集的[用户数，项目数]的得分矩阵中验证集交互过的项目的得分设置为0，以防验证集的项目造成干扰
            for i in range(len(val_data)):
                if val_data[i][0] >= start_index and val_data[i][0] < end_index:
                    row = val_data[i][0] - start_index
                    col = torch.LongTensor(list(val_data[i][1:])) - self.num_user
                    score_matrix_tt[row][col] = 1e-5
            # 将验证集的[用户数，项目数]的得分矩阵中测试集交互过的项目的得分设置为0，以防测试集的项目造成干扰
            for i in range(len(test_data)):
                if test_data[i][0] >= start_index and test_data[i][0] < end_index:
                    row = test_data[i][0] - start_index
                    col = torch.LongTensor(list(test_data[i][1:])) - self.num_user
                    score_matrix_vt[row][col] = 1e-5
            _, index_of_rank_list_vt = torch.topk(score_matrix_vt, topk)
            all_index_of_rank_list_vt = torch.cat(
                (all_index_of_rank_list_vt, index_of_rank_list_vt.cpu() + self.num_user),
                dim=0)
            _, index_of_rank_list_tt = torch.topk(score_matrix_tt, topk)
            all_index_of_rank_list_tt = torch.cat(
                (all_index_of_rank_list_tt, index_of_rank_list_tt.cpu() + self.num_user),
                dim=0)
            start_index = end_index

            if end_index + step < self.num_user:
                end_index += step
            else:
                end_index = self.num_user

        return all_index_of_rank_list_tra, all_index_of_rank_list_vt, all_index_of_rank_list_tt

    def accuracy(self, rank_list, topk=50):
        length = self.num_user
        precision_50 = recall_50 = ndcg_50 = 0.0
        precision_20 = recall_20 = ndcg_20 = 0.0
        precision_10 = recall_10 = ndcg_10 = 0.0
        precision_5 = recall_5 = ndcg_5 = 0.0
        precision_1 = recall_1 = ndcg_1 = 0.0
        for row, col in self.user_item_dict.items():
            # col = np.array(list(col))-self.num_user
            user = row
            pos_items = set(col)
            num_pos = len(pos_items)
            # 获取推荐结果的排序列表
            items_list_50 = rank_list[user].tolist()
            items_list_20 = items_list_50[:20]
            items_list_10 = items_list_50[:10]
            items_list_5 = items_list_50[:5]
            items_list_1 = items_list_50[:1]
            items_50 = set(items_list_50)
            items_20 = set(items_list_20)
            items_10 = set(items_list_10)
            items_5 = set(items_list_5)
            items_1 = set(items_list_1)
            #####top50#####
            num_hit_50 = len(pos_items.intersection(items_50))
            precision_50 += float(num_hit_50 / topk)
            recall_50 += float(num_hit_50 / num_pos)
            ndcg_score_50 = 0.0
            max_ndcg_score_50 = 0.0
            for i in range(min(num_hit_50, topk)):
                max_ndcg_score_50 += 1 / math.log2(i + 2)
            if max_ndcg_score_50 == 0:
                continue
            for i, temp_item in enumerate(items_list_50):
                if temp_item in pos_items:
                    ndcg_score_50 += 1 / math.log2(i + 2)
            ndcg_50 += ndcg_score_50 / max_ndcg_score_50
            ######top20#####
            num_hit_20 = len(pos_items.intersection(items_20))
            precision_20 += float(num_hit_20 / 20)
            recall_20 += float(num_hit_20 / num_pos)
            ndcg_score_20 = 0.0
            max_ndcg_score_20 = 0.0
            for i in range(min(num_hit_20, 20)):
                max_ndcg_score_20 += 1 / math.log2(i + 2)
            if max_ndcg_score_20 == 0:
                continue
            for i, temp_item in enumerate(items_list_20):
                if temp_item in pos_items:
                    ndcg_score_20 += 1 / math.log2(i + 2)
            ndcg_20 += ndcg_score_20 / max_ndcg_score_20
            ######top10######
            num_hit_10 = len(pos_items.intersection(items_10))
            precision_10 += float(num_hit_10 / 10)
            recall_10 += float(num_hit_10 / num_pos)
            ndcg_score_10 = 0.0
            max_ndcg_score_10 = 0.0
            for i in range(min(num_hit_10, 10)):
                max_ndcg_score_10 += 1 / math.log2(i + 2)
            if max_ndcg_score_10 == 0:
                continue
            for i, temp_item in enumerate(items_list_10):
                if temp_item in pos_items:
                    ndcg_score_10 += 1 / math.log2(i + 2)
            ndcg_10 += ndcg_score_10 / max_ndcg_score_10
            ######top5########
            num_hit_5 = len(pos_items.intersection(items_5))
            precision_5 += float(num_hit_5 / 5)
            recall_5 += float(num_hit_5 / num_pos)
            ndcg_score_5 = 0.0
            max_ndcg_score_5 = 0.0
            for i in range(min(num_hit_5, 5)):
                max_ndcg_score_5 += 1 / math.log2(i + 2)
            if max_ndcg_score_5 == 0:
                continue
            for i, temp_item in enumerate(items_list_5):
                if temp_item in pos_items:
                    ndcg_score_5 += 1 / math.log2(i + 2)
            ndcg_5 += ndcg_score_5 / max_ndcg_score_5
            #####top1#########
            num_hit_1 = len(pos_items.intersection(items_1))
            precision_1 += float(num_hit_1 / 1)
            recall_1 += float(num_hit_1 / num_pos)
            ndcg_score_1 = 0.0
            max_ndcg_score_1 = 0.0
            for i in range(min(num_hit_1, 1)):
                max_ndcg_score_1 += 1 / math.log2(i + 2)
            if max_ndcg_score_1 == 0:
                continue
            for i, temp_item in enumerate(items_list_1):
                if temp_item in pos_items:
                    ndcg_score_1 += 1 / math.log2(i + 2)
            ndcg_1 += ndcg_score_1 / max_ndcg_score_1

        return precision_50 / length, recall_50 / length, ndcg_50 / length, \
               precision_20 / length, recall_20 / length, ndcg_20 / length, \
               precision_10 / length, recall_10 / length, ndcg_10 / length, \
               precision_5 / length, recall_5 / length, ndcg_5 / length, \
               precision_1 / length, recall_1 / length, ndcg_1 / length

    def full_accuracy(self, val_data, rank_list, topk=50):
        length = len(val_data)
        precision_20 = recall_20 = ndcg_20 = 0.0
        precision_50 = recall_50 = ndcg_50 = 0.0
        precision_10 = recall_10 = ndcg_10 = 0.0
        precision_5 = recall_5 = ndcg_5 = 0.0
        precision_1 = recall_1 = ndcg_1 = 0.0
        count = 0
        # pdb.set_trace()
        for data in val_data:
            user = data[0]
            pos_i = data[1:]
            pos_temp = []
            # pdb.set_trace()
            if len(pos_i) == 0:
                length = length - 1
                count += 1
                continue

            for item in pos_i:
                pos_temp.append(item)
            pos_items = set(pos_temp)

            num_pos = len(pos_items)
            items_list_50 = rank_list[user].tolist()
            items_list_20 = items_list_50[:20]
            items_list_10 = items_list_50[:10]
            items_list_5 = items_list_50[:5]
            items_list_1 = items_list_50[:1]
            items_50 = set(items_list_50)
            items_20 = set(items_list_20)
            items_10 = set(items_list_10)
            items_5 = set(items_list_5)
            items_1 = set(items_list_1)
            ######top50############
            num_hit_50 = len(pos_items.intersection(items_50))
            precision_50 += float(num_hit_50 / topk)
            recall_50 += float(num_hit_50 / num_pos)
            ndcg_score_50 = 0.0
            max_ndcg_score_50 = 0.0
            for i in range(min(num_hit_50, topk)):
                max_ndcg_score_50 += 1 / math.log2(i + 2)
            if max_ndcg_score_50 == 0:
                continue
            for i, temp_item in enumerate(items_list_50):
                if temp_item in pos_items:
                    ndcg_score_50 += 1 / math.log2(i + 2)
            ndcg_50 += ndcg_score_50 / max_ndcg_score_50
            ########top20###############
            num_hit_20 = len(pos_items.intersection(items_20))
            precision_20 += float(num_hit_20 / 20)
            recall_20 += float(num_hit_20 / num_pos)
            ndcg_score_20 = 0.0
            max_ndcg_score_20 = 0.0
            for i in range(min(num_hit_20, 20)):
                max_ndcg_score_20 += 1 / math.log2(i + 2)
            if max_ndcg_score_20 == 0:
                continue
            for i, temp_item in enumerate(items_list_20):
                if temp_item in pos_items:
                    ndcg_score_20 += 1 / math.log2(i + 2)
            ndcg_20 += ndcg_score_20 / max_ndcg_score_20
            #######top10############
            num_hit_10 = len(pos_items.intersection(items_10))
            precision_10 += float(num_hit_10 / 10)
            recall_10 += float(num_hit_10 / num_pos)
            ndcg_score_10 = 0.0
            max_ndcg_score_10 = 0.0
            for i in range(min(num_hit_10, 10)):
                max_ndcg_score_10 += 1 / math.log2(i + 2)
            if max_ndcg_score_10 == 0:
                continue
            for i, temp_item in enumerate(items_list_10):
                if temp_item in pos_items:
                    ndcg_score_10 += 1 / math.log2(i + 2)
            ndcg_10 += ndcg_score_10 / max_ndcg_score_10

            num_hit_5 = len(pos_items.intersection(items_5))
            precision_5 += float(num_hit_5 / 5)
            recall_5 += float(num_hit_5 / num_pos)
            ndcg_score_5 = 0.0
            max_ndcg_score_5 = 0.0
            for i in range(min(num_hit_5, 5)):
                max_ndcg_score_5 += 1 / math.log2(i + 2)
            if max_ndcg_score_5 == 0:
                continue
            for i, temp_item in enumerate(items_list_5):
                if temp_item in pos_items:
                    ndcg_score_5 += 1 / math.log2(i + 2)
            ndcg_5 += ndcg_score_5 / max_ndcg_score_5

            num_hit_1 = len(pos_items.intersection(items_1))
            precision_1 += float(num_hit_1 / 1)
            recall_1 += float(num_hit_1 / num_pos)
            ndcg_score_1 = 0.0
            max_ndcg_score_1 = 0.0
            for i in range(min(num_hit_1, 1)):
                max_ndcg_score_1 += 1 / math.log2(i + 2)
            if max_ndcg_score_1 == 0:
                continue
            for i, temp_item in enumerate(items_list_1):
                if temp_item in pos_items:
                    ndcg_score_1 += 1 / math.log2(i + 2)
            ndcg_1 += ndcg_score_1 / max_ndcg_score_1
        return precision_50 / length, recall_50 / length, ndcg_50 / length, \
               precision_20 / length, recall_20 / length, ndcg_20 / length, \
               precision_10 / length, recall_10 / length, ndcg_10 / length, \
               precision_5 / length, recall_5 / length, ndcg_5 / length, \
               precision_1 / length, recall_1 / length, ndcg_1 / length

