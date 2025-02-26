import h5py
import numpy as np
import pandas as pd
from scipy import sparse
import scipy.sparse as sp
import tensorflow as tf
import random
#
# # Step 1: 读取.h5文件中的基因名称和索引
# file_path_h5 = '../2-dataset/mESC2/ST_t0.h5'
# # Step 2: 读取.txt文件中的基因对信息
# file_path_txt = '../2-dataset/DB_pairs_TF_gene/mesc2_gene_pairs_400.txt'
# npz_file_path = 'mESC2 adjacency_matrix.npz'
# try:
#     # 打开.h5文件并尝试读取数据
#     with h5py.File(file_path_h5, 'r') as f:
#         if 'STrans' in f:
#             dataset = f['STrans']
#             if 'axis0' in dataset:
#                 gene_names = dataset['axis0'][:]  # 从 'STrans/axis0' 读取数据
#                 gene_indices = {gene.decode('utf-8').lower(): idx for idx, gene in enumerate(gene_names)}
#                 print("Successfully read data from 'STrans/axis0' dataset.")
#             else:
#                 print("Dataset 'axis0' not found in 'STrans'.")
#         else:
#             print("Dataset 'STrans' not found in the .h5 file.")
# except Exception as e:
#     print(f"Error reading .h5 file: {str(e)}")
#
#
# gene_pairs = []
# with open(file_path_txt, 'r') as f:
#     for line in f:
#         parts = line.strip().split('\t')
#         if len(parts) == 3:
#             gene1 = parts[0]
#             gene2 = parts[1]
#             relation = int(parts[2])  # 将第三列转换为整数，0或1
#             gene_pairs.append((gene1, gene2, relation))
#
# # Step 3: 更新邻接矩阵
# num_genes = len(gene_names)
# adj_matrix = np.zeros((num_genes, num_genes), dtype=int)
#
# for gene1, gene2, relation in gene_pairs:
#     if gene1 in gene_indices and gene2 in gene_indices:
#         idx1 = gene_indices[gene1]
#         idx2 = gene_indices[gene2]
#         if relation == 1:
#             adj_matrix[idx1, idx2] = 1
#
# # Step 4: 转换为CSR格式的稀疏矩阵并保存为.npz文件
# csr_adj_matrix = sparse.csr_matrix(adj_matrix)
# sparse.save_npz(npz_file_path, csr_adj_matrix)
#
# # Counting the number of 1s in the adjacency matrix
# num_ones = np.sum(adj_matrix == 1)
# print(f"Number of 1s in the adjacency matrix: {num_ones}")
#
# print(f"CSR格式的邻接矩阵已保存为.npz文件：{npz_file_path}")

import h5py

# # 读取.h5文件
# h5_file_path = '../2-dataset/sim2/ST_t0.h5'  # 替换为你的.h5文件路径
# gene_list_path = '../2-dataset/sim2_geneName_map.txt'  # 指定保存基因名称的txt文件路径
#
# # 打开.h5文件
# with h5py.File(h5_file_path, 'r') as f:
#     # 获取STrans下的axis0数据
#     axis0_data = f['STrans']['axis0'][:]
#     axis0_data = axis0_data.astype(str)  # 将数据转换为字符串类型
#     axis0_data = [gene.lower() for gene in axis0_data]  # 将基因名称转换为小写
#
# # 写入基因名称到txt文件
# with open(gene_list_path, 'w') as file:
#     for gene in axis0_data:
#         file.write(f"{gene}\t{gene}\n")  # 将每个基因写入txt文件，格式为 "gene\tgene"
#
# print(f"基因名称已经保存到 {gene_list_path}")


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape



def mask_test_edges_general_link_prediction_2(adj, test_percent=10., val_percent=10.):
    """
    Task 1: General Directed Link Prediction: get Train/Validation/Test

    :param adj: complete sparse adjacency matrix of the graph
    :param test_percent: percentage of edges in test set
    :param val_percent: percentage of edges in validation set
    :return: train incomplete adjacency matrix, validation and test sets
    """

    # Remove diagonal elements of adjacency matrix 移除邻接矩阵的对角元素
    adj = adj - sp.dia_matrix((adj.diagonal()[None, :], [0]), shape = adj.shape)
    adj.eliminate_zeros()                                # 将稀疏矩阵中的零元素去除的方法
    edges_positive, _, _ = sparse_to_tuple(adj)
    # edges_positive 包含了矩阵中非零元素的行、列索引和对应的值。另外两个下划线 _ 表示忽略了额外的返回值。
    # Number of positive (and negative) edges in test and val sets:
    num_test = int(np.floor(edges_positive.shape[0] / (100. / test_percent)))
    num_val = int(np.floor(edges_positive.shape[0] / (100. / val_percent)))

    # Sample positive edges for test and val sets:
    edges_positive_idx = np.arange(edges_positive.shape[0])
    np.random.shuffle(edges_positive_idx)
    val_edge_idx = edges_positive_idx[:num_val]
    # positive val edges
    val_edges = edges_positive[val_edge_idx]
    test_edge_idx = edges_positive_idx[num_val:(num_val + num_test)]
    # positive test edges
    test_edges = edges_positive[test_edge_idx]
    # positive train edges
    train_edges = np.delete(edges_positive, np.hstack([test_edge_idx, val_edge_idx]), axis = 0)

    # (Text from philipjackson)
    # The above strategy for sampling without replacement will not work for sampling
    # negative edges on large graphs, because the pool of negative edges
    # is much much larger due to sparsity, therefore we'll use the following strategy:
    # 1. sample random linear indices from adjacency matrix WITH REPLACEMENT 1.从具有替换的邻接矩阵中采样随机线性索引
    # (without replacement is super slow). sample more than we need so we'll probably
    # have enough after all the filtering steps.
    # 2. remove any edges that have already been added to the other edge lists  删除已添加到其他边列表中的所有边
    # 3. convert to (i,j) coordinates
    # 4. remove any duplicate elements if there are any
    # 5. remove any diagonal elements
    # 6. if we don't have enough edges, repeat this process until we get enough
    positive_idx, _, _ = sparse_to_tuple(adj) # positive_idx 是一个包含稀疏矩阵 adj 中所有非零元素的行索引和列索引的二维数组。 [i,j] coord pairs for all true edges
    positive_idx = positive_idx[:,0]*adj.shape[0] + positive_idx[:,1] # linear indices  positive_idx[:,0]行索引/列索引
    # Test set
    test_edges_false = np.empty((0,2),dtype='int64')
    idx_test_edges_false = np.empty((0,),dtype='int64')
    while len(test_edges_false) < len(test_edges):
        # step 1:
        idx = np.random.choice(adj.shape[0]**2, 2*(num_test - len(test_edges_false)), replace = True)
        # step 2:
        idx = idx[~np.in1d(idx, positive_idx, assume_unique = True)]
        idx = idx[~np.in1d(idx, idx_test_edges_false, assume_unique = True)]
        # step 3:
        rowidx = idx // adj.shape[0]
        colidx = idx % adj.shape[0]
        coords = np.vstack((rowidx, colidx)).transpose()
        # step 4:
        coords = np.unique(coords, axis=0)
        np.random.shuffle(coords)
        # step 5:
        coords = coords[coords[:,0] != coords[:,1]]
        # step 6:
        coords = coords[:min(num_test, len(idx))]
        test_edges_false = np.append(test_edges_false, coords, axis = 0)
        idx = idx[:min(num_test, len(idx))]
        idx_test_edges_false = np.append(idx_test_edges_false, idx)

    # Validation set
    val_edges_false = np.empty((0,2), dtype = 'int64')
    idx_val_edges_false = np.empty((0,), dtype = 'int64')
    while len(val_edges_false) < len(val_edges):
        # step 1:
        idx = np.random.choice(adj.shape[0]**2, 2*(num_val - len(val_edges_false)), replace = True)
        # step 2:
        idx = idx[~np.in1d(idx, positive_idx, assume_unique = True)]
        idx = idx[~np.in1d(idx, idx_test_edges_false, assume_unique = True)]
        idx = idx[~np.in1d(idx, idx_val_edges_false, assume_unique = True)]
        # step 3:
        rowidx = idx // adj.shape[0]
        colidx = idx % adj.shape[0]
        coords = np.vstack((rowidx, colidx)).transpose()
        # step 4:
        coords = np.unique(coords, axis = 0)
        np.random.shuffle(coords)
        # step 5:
        coords = coords[coords[:,0] != coords[:,1]]
        # step 6:
        coords = coords[:min(num_val, len(idx))]
        val_edges_false = np.append(val_edges_false, coords, axis=0)
        idx = idx[:min(num_val, len(idx))]
        idx_val_edges_false = np.append(idx_val_edges_false, idx)

        # Combine validation and test edges
    all_edges = np.vstack([val_edges, test_edges])
    all_edges_false = np.vstack([val_edges_false, test_edges_false])

    # Sanity checks:
    train_edges_linear = train_edges[:,0]*adj.shape[0] + train_edges[:,1]
    test_edges_linear = test_edges[:,0]*adj.shape[0] + test_edges[:,1]
    assert not np.any(np.in1d(idx_test_edges_false, positive_idx))
    assert not np.any(np.in1d(idx_val_edges_false, positive_idx))
    assert not np.any(np.in1d(val_edges[:,0]*adj.shape[0] + val_edges[:,1], train_edges_linear))
    assert not np.any(np.in1d(test_edges_linear, train_edges_linear))
    assert not np.any(np.in1d(val_edges[:,0]*adj.shape[0] + val_edges[:,1], test_edges_linear))

    # Re-build train adjacency matrix
    data = np.ones(train_edges.shape[0])
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape = adj.shape)

    return adj_train, all_edges, all_edges_false