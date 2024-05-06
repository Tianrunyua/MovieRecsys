'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='Seed init.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--model_name', default='MFMGNN', help='Model name.')
    parser.add_argument('--data_path', default='./Mfmgnn_iptvrs/Data/', help='Dataset path')
    parser.add_argument('--PATH_weight_load', default=None, help='Loading weight filename.')
    parser.add_argument('--PATH_weight_save', default=None, help='Writing weight filename.')
    parser.add_argument('--l_r', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay.')
    parser.add_argument('--weight_decay_adm', type=float, default=0, help='Weight decay.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--dim_latent', type=int, default=128, help='Latent dimension.')
    parser.add_argument('--num_epoch', type=int, default=30, help='Epoch number.')
    parser.add_argument('--num_workers', type=int, default=0, help='Workers number.')
    parser.add_argument('--aggr_mode', default='add', help='Aggregation mode.')
    parser.add_argument('--user_aggr_mode', default='softmax', help='Aggregation mode.')
    parser.add_argument('--construction', default='weighted_sum', help='information construction weighted_sum')
    parser.add_argument('--num_layer', type=int, default=1, help='Layer number.')
    parser.add_argument('--has_id', type=bool, default=True, help='Has id_embedding')
    parser.add_argument('--dataset', default='IPTV', help='Dataset path')
    parser.add_argument('--varient', default='random', help='model varient')
    parser.add_argument('--sampling_user', type=int, default=10, help='user co-occurance number')
    parser.add_argument('--sampling_item_sim_t', type=int, default=15, help='item co-occurance number')
    parser.add_argument('--sampling_item_sim_v', type=int, default=15, help='item co-occurance number')
    parser.add_argument('--sampling_item_com', type=int, default=20, help='item co-occurance number')
    parser.add_argument('--itemgraph_layer', type=int, default=1,help='Number of item graph conv layers')
    parser.add_argument('--pre_num', type=int, default=6,help='Number of feature preference')
    parser.add_argument('--num_user', type=int, default=143916, help='User number')
    # parser.add_argument('--num_user', type=int, default=100, help='User number')
    parser.add_argument('--num_item', type=int, default=6259, help='Item number')
    parser.add_argument('--missrate', type=float, default=0.0, help='modality miss rate')
    return parser.parse_args()
