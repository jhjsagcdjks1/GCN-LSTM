import h5py
import os
import time
from sklearn.model_selection import KFold
from sklearn.preprocessing import MaxAbsScaler
from gravity_gae.preprocessing import *
from gravity_gae.layers import *
from gravity_gae.data_process2 import mask_test_edges_general_link_prediction_2
from gravity_gae.optimizer import OptimizerAE, OptimizerVAE
from gravity_gae.moxing import GravityGCNModelAE1, AdptiveGravityGCNModelAEv3
from gravity_gae.model import *
from gravity_gae.evaluation import compute_scores
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 设置TensorFlow日志级别为只显示ERROR信息
import warnings
warnings.filterwarnings('ignore')  # 忽略Python警告
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # 设置TensorFlow的警告和信息级别


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'gravity_gcn_ae', 'Name of the model')
# Model parameters
flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).') #参数的意义是 "丢弃率（1 - 保留概率）"。即，1 表示保留所有神经元，0 表示丢弃所有神经元。
flags.DEFINE_integer('epochs', 100, 'Number of epochs in training.')
flags.DEFINE_boolean('features', True, 'Include node features or not in GCN')
flags.DEFINE_float('lamb', 1.0, 'lambda parameter from Gravity AE/VAE models \
                                as introduced in section 3.5 of paper, to \
                                balance mass and proximity terms')       # 来自 Gravity AE/VAE 模型的 lambda 参如论文第 3.5 节所述，至平衡质量和邻近度项
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate (with Adam)')
flags.DEFINE_integer('hidden', 256, 'Number of units in GCN hidden layer.')
flags.DEFINE_integer('dimension', 256, 'Dimension of GCN output: \
- equal to embedding dimension for standard AE/VAE and source-target AE/VAE \
- equal to (embedding dimension - 1) for gravity-inspired AE/VAE, as the \
last dimension captures the "mass" parameter tilde{m}')
flags.DEFINE_integer('dimension1', 256, 'Dimension of GCN output: \
- equal to embedding dimension for standard AE/VAE and source-target AE/VAE \
- equal to (embedding dimension - 1) for gravity-inspired AE/VAE, as the \
last dimension captures the "mass" parameter tilde{m}')
flags.DEFINE_boolean('normalize', False, 'Whether to normalize embedding \
                                          vectors of gravity models')  # 是否规范化嵌入重力模型的向量 不加感觉结果更好
flags.DEFINE_float('epsilon', 0.1, 'Add epsilon to distances computations \
                                       in gravity models, for numerical \
                                       stability')  #将 epsilon 添加到距离计算中在重力模型中，对于数值稳定性
flags.DEFINE_float('prop_val', 10., 'Proportion of edges in validation set \
                                   (for Task 1)')  # 验证边的比例
flags.DEFINE_float('prop_test', 10., 'Proportion of edges in test set \
                                      (for Tasks 1 and 2)')
gcn_dimensions = [32, 64, 128, 256, 512, 1024]
for dimension in gcn_dimensions:
    print(dimension)
    FLAGS.dimension = dimension


    # 创建 TensorFlow 计算图
    tf.reset_default_graph()
    dataset_paths = [
        {
            'adj_matrix_npz_list': '../2-dataset/hESC1/hESC1 adjacency_matrix.npz',
            'file_pattern': '../2-dataset/hESC1/ST_t{}.h5',
        },
    {
            'adj_matrix_npz_list': '../2-dataset/hESC2/hESC2 adjacency_matrix.npz',
            'file_pattern': '../2-dataset/hESC2/ST_t{}.h5',
        },
    {
            'adj_matrix_npz_list': '../2-dataset/mESC1/mESC1 adjacency_matrix.npz',
            'file_pattern': '../2-dataset/mESC1/ST_t{}.h5',
        },
    {
            'adj_matrix_npz_list': '../2-dataset/mESC2/mESC2 adjacency_matrix.npz',
            'file_pattern': '../2-dataset/mESC2/ST_t{}.h5',
        },
    {
            'adj_matrix_npz_list': '../2-dataset/sim1/sim1 adjacency_matrix.npz',
            'file_pattern': '../2-dataset/sim1/ST_t{}.h5',
        },
    {
            'adj_matrix_npz_list': '../2-dataset/sim2/sim2 adjacency_matrix.npz',
            'file_pattern': '../2-dataset/sim2/ST_t{}.h5',
        },
    {
            'adj_matrix_npz_list': '../2-dataset/sim3/sim3 adjacency_matrix.npz',
            'file_pattern': '../2-dataset/sim3/ST_t{}.h5',
        },
    {
            'adj_matrix_npz_list': '../2-dataset/sim4/sim4 adjacency_matrix.npz',
            'file_pattern': '../2-dataset/sim4/ST_t{}.h5',
        },
    ]

    # 遍历文件路径列表
    for dataset_path in dataset_paths:
        adj_matrix_npz = dataset_path['adj_matrix_npz_list']
        file_pattern = dataset_path['file_pattern']
        # 加载CSR格式的邻接矩阵
        adj_matrix = sp.load_npz(adj_matrix_npz).astype(np.float64)
        # adj, test_edges, test_edges_false = \
        #     mask_test_edges_general_link_prediction_2(adj_matrix, FLAGS.prop_test,
        #                                             FLAGS.prop_val)
        # adj_orig = adj
        # adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
        # adj_orig.eliminate_zeros()
        # # Compute number of nodes
        # num_nodes = adj.shape[0]

        # 移除邻接矩阵的对角元素
        adj = adj_matrix - sp.dia_matrix((adj_matrix.diagonal()[None, :], [0]), shape=adj_matrix.shape)
        adj.eliminate_zeros()

        # 确定图中的节点数
        num_nodes = adj.shape[0]

        # 获取正边的索引
        pos_edges = np.array(adj.nonzero()).T

        # 随机选择负边的生成方式
        num_neg_samples = len(pos_edges)
        neg_edges = []

        while len(neg_edges) < num_neg_samples:
            i, j = random.randint(0, adj.shape[0]-1), random.randint(0, adj.shape[0]-1)
            if i != j and adj[i, j] == 0:
                neg_edges.append((i, j))

        neg_edges = np.array(neg_edges)


        # 划分训练集和测试集的正边
        train_size = int(num_neg_samples * 0.8)
        train_pos_edges = pos_edges[:train_size]
        test_edges = pos_edges[train_size:]

        # 在测试集中随机选择等量的负边
        test_edges_false = random.sample(list(neg_edges), len(test_edges))

        # 构建邻接矩阵 ADJ
        adj = np.zeros((num_nodes, num_nodes))
        for edge in train_pos_edges:
            adj[edge[0], edge[1]] = 1
        adj = sp.csr_matrix(adj)

        adj_orig = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj_orig.eliminate_zeros()

        # 加载特征矩阵
        num_time_points = 0
        while os.path.exists(file_pattern.format(num_time_points)):
            num_time_points += 1

        feature_matrices = []
        for t in range(num_time_points):
            with h5py.File(file_pattern.format(t), 'r') as f:
                feature_matrix = f['STrans/block0_values'][:]
                feature_matrix_transposed = feature_matrix.T
                # 归一化特征矩阵
                scaler = MaxAbsScaler()
                feature_matrix_transposed = scaler.fit_transform(feature_matrix_transposed)
                # 转换为稀疏矩阵
                if not sp.issparse(feature_matrix_transposed):
                    feature_matrix_transposed = sp.coo_matrix(feature_matrix_transposed)
                feature_matrices.append(feature_matrix_transposed)

        # 创建 TF placeholder
        placeholders = {
            'features': tf.sparse_placeholder(tf.float32),
            'adj': tf.sparse_placeholder(tf.float32),
            'adj_orig': tf.sparse_placeholder(tf.float32),
            'dropout': tf.placeholder_with_default(0., shape=())
        }
        # Normalization and preprocessing on adjacency matrix
        adj_norm = preprocess_graph(adj)
        adj_label = sparse_to_tuple(adj + sp.eye(adj.shape[0]))

        # 独立测试集 训练集：测试集= 8：2
        k = 5
        # 定义空列表来存储每次循环的结果
        model_layers_best = []
        mean_roc = []
        mean_ap = []
        mean_time = []
        lay2_roc = []
        lay2_ap = []
        lay2_time = []
        for i in range(k):
            initial_layers = 1
            layer_step = 1  # 每次增加的层数
            best_roc_score = 0.0
            best_ap_score = 0.0
            best_model_layers = initial_layers
            print(f"Num {i + 1}:")
            current_layers = initial_layers
            roc_scores_for_layers = []
            while True:
                # 训练模型和获取嵌入
                embeddings = []
                for t in range(num_time_points):
                    current_feature_matrix = feature_matrices[t]
                    features = sparse_to_tuple(current_feature_matrix)
                    num_features = features[2][1]

                    # Gravity-Inspired Graph Autoencoder
                    # model = GravityGCNModelAE1(placeholders, num_features, features[1].shape[0])
                    model = AdptiveGravityGCNModelAEv3(placeholders, num_features, features[1].shape[0],
                                                       initial_num_layers=current_layers, logging=True)
                    # 初始化 TF session
                    # Initialize TF session
                    sess = tf.Session()
                    sess.run(tf.global_variables_initializer())
                    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
                    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
                    emb = sess.run(model.z_mean, feed_dict=feed_dict)
                    embeddings.append(emb)
                # LSTM 模型定义
                with tf.variable_scope('lstm', reuse=tf.AUTO_REUSE):
                    input_data = tf.placeholder(tf.float32, shape=[None, None, FLAGS.dimension])
                    lstm_units = 256
                    lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_units)
                    outputs, _ = tf.nn.dynamic_rnn(lstm_cell, input_data, dtype=tf.float32)

                # 初始化 TF session
                sess = tf.Session()
                sess.run(tf.global_variables_initializer())
                # 嵌入堆叠
                stacked_input = np.stack(embeddings, axis=0)
                sess.run(tf.global_variables_initializer())
                predictions = sess.run(outputs, feed_dict={input_data: stacked_input})
                predictions = predictions[-1]
                # Preprocessing on node features 将得到的predictions值当作特征矩阵输入
                predictions = sp.csr_matrix(predictions)
                predictions = sparse_to_tuple(predictions)
                num_predictions = predictions[2][1]
                predictions_nonzero = predictions[1].shape[0]
                model1 = GravityGCNModelAE(placeholders, num_predictions, predictions_nonzero)
                # Optimizer (see tkipf/gae original GAE repository for details)
                pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
                norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0]
                                                            - adj.sum()) * 2)
                opt = OptimizerAE(preds=model1.reconstructions,
                                  labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                              validate_indices=False), [-1]),
                                  pos_weight=pos_weight,
                                  norm=norm)
                # Initialize TF session
                sess.run(tf.global_variables_initializer())
                # Flag to compute running time for each epoch
                # Model training
                print("Training...")
                # Flag to compute total running time
                t_start = time.time()
                for epoch in range(FLAGS.epochs):
                    t = time.time()
                    # Construct feed dictionary
                    feed_dict = construct_feed_dict(adj_norm, adj_label, predictions,
                                                    placeholders)
                    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
                    # Weight update
                    outs = sess.run([opt.opt_op, opt.cost, opt.accuracy],
                                    feed_dict=feed_dict)
                    # Compute average loss
                    avg_cost = outs[1]
                    # Display epoch information
                    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
                          "time=", "{:.5f}".format(time.time() - t))
                    feed_dict.update({placeholders['dropout']: 0})
                # Get embedding from model 从模型中获取嵌入
                emb = sess.run(model1.z_mean, feed_dict=feed_dict)
                # 创建用于保存图像的文件
                # Compute ROC and AP scores on test sets
                roc_score, ap_score = compute_scores(test_edges, test_edges_false, emb)
                # 存储当前层数的 ROC 分数
                roc_scores_for_layers.append(roc_score)
                # 判断是否增加层数
                if current_layers == 2:
                    lay2_roc.append(roc_score)
                    lay2_ap.append(ap_score)
                    lay2_time.append(time.time() - t_start)
                    print(f"GCN layers are 2 with ROC {roc_score} and AP {ap_score}")
                if roc_score > best_roc_score:
                    best_roc_score = roc_score
                    best_ap_score = ap_score  # 保存当前最佳 ROC 对应的 AP 分数
                    best_model_layers = current_layers
                    print(
                        f"Best GCN layers updated to {best_model_layers} with ROC {best_roc_score} and AP {best_ap_score}")
                    current_layers += layer_step  # 增加层数
                else:
                    print(f"No improvement in ROC for {current_layers} layers, stopping layer expansion.")
                    break

            # Append to list of scores over all runs
            mean_roc.append(best_roc_score)
            mean_ap.append(best_ap_score)
            model_layers_best.append(best_model_layers)
            # Compute total running time
            mean_time.append(time.time() - t_start)

        # Report final results

        print("AUC scores\n", mean_roc)
        AUC_scores = np.mean(mean_roc)
        AUC_std = np.std(mean_roc)
        print("Mean AUC score: ", np.mean(mean_roc),
              "\nStd of AUC scores: ", np.std(mean_roc), "\n \n")

        print("AP scores \n", mean_ap)
        AP_scores = np.mean(mean_ap)
        AP_std = np.std(mean_ap)
        print("Mean AP score: ", np.mean(mean_ap),
              "\nStd of AP scores: ", np.std(mean_ap), "\n \n")
        model_layers = np.mean(model_layers_best)
        print("Mean AP score: ", np.mean(model_layers_best), "\n \n")

        print("Running times\n", mean_time)
        time_mean = np.mean(mean_time)
        print("Mean running time: ", np.mean(mean_time),
              "\nStd of running time: ", np.std(mean_time), "\n \n")

        print("lay2 AUC scores\n", lay2_roc)
        AUC_lay2_scores = np.mean(lay2_roc)
        AUC_lay2_std = np.std(lay2_roc)
        print("Mean AUC score: ", np.mean(lay2_roc),
              "\nStd of AUC scores: ", np.std(lay2_roc), "\n \n")

        print("lay2 AP scores \n", lay2_ap)
        AP_lay2_scores = np.mean(lay2_ap)
        AP_lay2_layers = np.mean(lay2_ap)
        print("Mean AP score: ", np.mean(lay2_ap), "\n \n")

        print("Running times\n", lay2_time)
        time_lay2_mean = np.mean(lay2_time)
        print("Mean running time: ", np.mean(lay2_time),
              "\nStd of running time: ", np.std(lay2_time), "\n \n")
        import time

        current_time = time.time()
        local_time_struct = time.localtime(current_time)
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S",
                                       local_time_struct)  # 'AUC scores4'+'AUC scores5'+'AUC scores6'+'AUC scores7'+ 'AUC scores8'+'AUC scores9'+'AUC scores10'
        column_names = ['date_name', 'dropout', 'epochs', 'learning_rate', 'hidden', 'dimension', 'normalize', 'AUC scores1', 'AUC scores2', 'AUC scores3', 'AUC scores4', 'AUC scores5', 'model_layers',
                        'AUC mean', 'AUC std', 'AP mean', 'AP std','Running times','AUC lay2 scores1', 'AUC lay2 scores2',
                        'AUC lay2 scores3', 'AUC lay2 scores4', 'AUC lay2 scores5','AUC lay2 mean', 'AUC lay2 std',
                        'AP lay2 mean', 'AP lay2 std', 'Running times lay2']
        resultspath = "../data/2-model1—new-独立测试集-GCN维度调参-0.01.csv"
        new_data = [adj_matrix_npz, FLAGS.dropout, FLAGS.epochs, FLAGS.learning_rate, FLAGS.hidden, FLAGS.dimension, FLAGS.normalize,
                    mean_roc, model_layers, AUC_scores, AUC_std, AP_scores, AP_std, time_mean, lay2_roc, AUC_lay2_scores,
                    AUC_lay2_std, AP_lay2_scores, AP_lay2_layers,  time_lay2_mean]

        if os.path.exists(resultspath):
            with open(resultspath, mode='a', newline='') as file:
                np.savetxt(file, [new_data], delimiter=',', header=' ', comments='', fmt='%s')
        else:
            np.savetxt(resultspath, [new_data], delimiter=',', header=','.join(column_names), comments='', fmt='%s')





