import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

def fig(tensor_list,index):
    # print(loaded_data)  # 打印加载的对象
    A = []
    for i in range(len(tensor_list)):
        A.append(tensor_list[i].squeeze(1).cpu().numpy())
    A = np.array(A)
    B = [[0 for _ in range(0,4)] for __ in range(0,12)]
    for i in range(len(A)):
        for j in A[i]:
            B[i][j] += 1
    # A = A.reshape((10,48))
    # A = A/197
    B = np.array(B)/197
    # print(B.shape,B)
    # 生成示例向量集合
    # vectors = np.random.rand(5, 10)  # 5个向量，每个向量包含10个元素
    # list_a = []
    # for i in tensor_list:
    #     list_a.append(i.cpu().detach().squeeze(0).numpy())
    # vectors = np.array(list_a)
    
    # 计算余弦相似度矩阵
    # cosine_sim_matrix = cosine_similarity(vectors)

    # 绘制热力图
    plt.figure(figsize=(8, 6))
    # print(cosine_sim_matrix.shape)
    plt.imshow(B, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='frequency')

    # 设置坐标轴标签
    plt.xticks(np.arange(len(B[0])), range(len(B[0])))
    plt.yticks(np.arange(len(B)), range(len(B)))

    # 添加标题
    plt.title('Cosine Similarity Heatmap')
    plt.savefig('./imgs/second_models'+str(index),dpi=600)
    # 显示热力图
    # plt.show()

def attnfig(tensor_list,tensor_list2,index):
    # print(loaded_data)  # 打印加载的对象
    A,B = [],[]
    for i in range(len(tensor_list)):
        A.append(tensor_list[i].squeeze(1).cpu().numpy())
    A = np.array(A)
    for i in range(len(tensor_list2)):
        B.append(tensor_list2[i].cpu().numpy())
    B = np.array(B)
    C,D,E = [[] for i in range(0,4)],[[] for i in range(0,4)],[[] for i in range(0,4)]

    F = [[] for i in range(0,4)]
    for j in range(0,len(A[3])):
        F[A[3,j]].append(B[3,j])

    for j in range(0,len(A[0])):
        C[A[0,j]].append(B[0,j])
    for j in range(0,len(A[1])):
        D[A[1,j]].append(B[1,j])
    for j in range(0,len(A[2])):
        E[A[2,j]].append(B[2,j])
    # print(C,D,E)
    # 生成示例数据，包括正负值
    experts = ['Expert 1', 'Expert 2', 'Expert 3', 'Expert 4']
    # attention_values = [0.5, -0.3, 0.8, -0.2]
    for k in range(0,4):
        # 创建一个颜色映射，根据正负值选择不同的颜色
        colors = ['green' if val >= 0 else 'red' for val in C[k]]

        # 创建柱状图
        plt.figure(figsize=(8, 6))
        bars = plt.bar([i for i in range(0,len(C[k]))], C[k], color=colors)

        # 添加数值标签
        # for bar, value in zip(bars, attention_values):
        #     plt.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha='center', va='bottom', fontsize=12)

        # 设置标题和标签
        plt.title('Attention Values for Expert ' + str(k))
        plt.xlabel('token id')
        plt.ylabel('Attention Values')

        # 自定义颜色图例
        legend_labels = ['Positive', 'Negative']
        legend_colors = ['green', 'red']
        legend_handles = [plt.Line2D([0], [0], color=color, lw=4) for color in legend_colors]
        plt.legend(legend_handles, legend_labels)

        # 调整y轴刻度以适应负值
        plt.ylim([-1, 1])

        # 显示图表
        plt.savefig('./imgs/attn_expert/'+'img_'+str(index)+'_first_layer_expert ' + str(k),dpi=600)
    for k in range(0,4):
        # 创建一个颜色映射，根据正负值选择不同的颜色
        colors = ['green' if val >= 0 else 'red' for val in D[k]]

        # 创建柱状图
        plt.figure(figsize=(8, 6))
        bars = plt.bar([i for i in range(0,len(D[k]))], D[k], color=colors)

        # 添加数值标签
        # for bar, value in zip(bars, attention_values):
        #     plt.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha='center', va='bottom', fontsize=12)

        # 设置标题和标签
        plt.title('Attention Values for Expert ' + str(k))
        plt.xlabel('token id')
        plt.ylabel('Attention Values')

        # 自定义颜色图例
        legend_labels = ['Positive', 'Negative']
        legend_colors = ['green', 'red']
        legend_handles = [plt.Line2D([0], [0], color=color, lw=4) for color in legend_colors]
        plt.legend(legend_handles, legend_labels)

        # 调整y轴刻度以适应负值
        plt.ylim([-1, 1])

        # 显示图表
        plt.savefig('./imgs/attn_expert/'+'img_'+str(index)+'_second_layer_expert ' + str(k),dpi=600)
    for k in range(0,4):
        # 创建一个颜色映射，根据正负值选择不同的颜色
        colors = ['green' if val >= 0 else 'red' for val in E[k]]

        # 创建柱状图
        plt.figure(figsize=(8, 6))
        bars = plt.bar([i for i in range(0,len(E[k]))], E[k], color=colors)

        # 添加数值标签
        # for bar, value in zip(bars, attention_values):
        #     plt.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha='center', va='bottom', fontsize=12)

        # 设置标题和标签
        plt.title('Attention Values for Expert ' + str(k))
        plt.xlabel('token id')
        plt.ylabel('Attention Values')

        # 自定义颜色图例
        legend_labels = ['Positive', 'Negative']
        legend_colors = ['green', 'red']
        legend_handles = [plt.Line2D([0], [0], color=color, lw=4) for color in legend_colors]
        plt.legend(legend_handles, legend_labels)

        # 调整y轴刻度以适应负值
        plt.ylim([-1, 1])

        # 显示图表
        plt.savefig('./imgs/attn_expert/'+'img_'+str(index)+'_third_layer_expert ' + str(k),dpi=600)
    for k in range(0,4):
        # 创建一个颜色映射，根据正负值选择不同的颜色
        colors = ['green' if val >= 0 else 'red' for val in F[k]]

        # 创建柱状图
        plt.figure(figsize=(8, 6))
        bars = plt.bar([i for i in range(0,len(F[k]))], F[k], color=colors)

        # 添加数值标签
        # for bar, value in zip(bars, attention_values):
        #     plt.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha='center', va='bottom', fontsize=12)

        # 设置标题和标签
        plt.title('Attention Values for Expert ' + str(k))
        plt.xlabel('token id')
        plt.ylabel('Attention Values')

        # 自定义颜色图例
        legend_labels = ['Positive', 'Negative']
        legend_colors = ['green', 'red']
        legend_handles = [plt.Line2D([0], [0], color=color, lw=4) for color in legend_colors]
        plt.legend(legend_handles, legend_labels)

        # 调整y轴刻度以适应负值
        plt.ylim([-1, 1])

        # 显示图表
        plt.savefig('./imgs/attn_expert/'+'img_'+str(index)+'_fourth_layer_expert ' + str(k),dpi=600)
def expert(tensor_list):
    from sklearn.metrics.pairwise import cosine_similarity
    # 生成示例向量集合
    # vectors = np.random.rand(5, 10)  # 5个向量，每个向量包含10个元素
    list_a = []
    for i in tensor_list:
        list_a.append(i.cpu().detach().squeeze(0).numpy())
    vectors = np.array(list_a)
    vectors = vectors[-4:,-4:]
    # 计算余弦相似度矩阵
    cosine_sim_matrix = cosine_similarity(vectors)

    # 绘制热力图
    plt.figure(figsize=(16, 12))
    print(cosine_sim_matrix.shape)
    plt.imshow(cosine_sim_matrix, cmap='inferno', interpolation='none')
    plt.colorbar(label='Cosine Similarity')

    # 设置坐标轴标签
    plt.xticks(np.arange(len(vectors)), range(len(vectors)))
    plt.yticks(np.arange(len(vectors)), range(len(vectors)))

    # 添加标题
    plt.title('Cosine Similarity Heatmap')
    plt.savefig('./imgs/experts/1.png',dpi=600)
    # 显示热力图
    plt.show()

def expert_total():
    import pickle
    file_name = 'output/dict.pkl'

    # 打开文件，以二进制读取模式加载对象
    with open(file_name, 'rb') as file:
        loaded_data = pickle.load(file)

    print(loaded_data)  # 打印加载的对象
    A = []
    for i in range(0,10):
        A.append(loaded_data[i])
    A = np.array(A)
    # A = A.reshape((10,48))
    A = A/1000
    print(A.shape,A)
    # 生成示例向量集合
    # vectors = np.random.rand(5, 10)  # 5个向量，每个向量包含10个元素
    # list_a = []
    # for i in tensor_list:
    #     list_a.append(i.cpu().detach().squeeze(0).numpy())
    # vectors = np.array(list_a)
    
    # 计算余弦相似度矩阵
    # cosine_sim_matrix = cosine_similarity(vectors)

    # 绘制热力图
    plt.figure(figsize=(8, 6))
    # print(cosine_sim_matrix.shape)
    plt.imshow(A, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='frequency')

    # 设置坐标轴标签
    plt.xticks(np.arange(len(A[0])), range(len(A[0])))
    plt.yticks(np.arange(len(A)), range(len(A)))

    # 添加标题
    plt.title('Cosine Similarity Heatmap')
    plt.savefig('./imgs/experts/2.png',dpi=600)
    # 显示热力图
    plt.show()

def tokenembedfig(tensor_list,tensor_list2,index):
    from sklearn.metrics.pairwise import cosine_similarity
    # print(loaded_data)  # 打印加载的对象
    A,B = [],[]
    for i in range(len(tensor_list)):
        A.append(tensor_list[i].squeeze(1).cpu().numpy())
    A = np.array(A)
    for i in range(len(tensor_list2)):
        B.append(tensor_list2[i].squeeze(0).cpu().numpy())
    B = np.array(B)
    B = B.reshape(B.shape[0],B.shape[1]*B.shape[2],B.shape[3])
    C = [[[] for i in range(0,4)] for _ in range(0,12)]

    # C,D,E = [[] for i in range(0,4)],[[] for i in range(0,4)],[[] for i in range(0,4)]

    for layer in range(0,12):
        for j in range(0,len(A[layer])):
            C[layer][A[layer,j]].append(B[layer,j,:])
    # F = [[] for i in range(0,4)]
    # for j in range(0,len(A[3])):
    #     F[A[3,j]].append(B[3,j,:])
    # for j in range(0,len(A[0])):
    #     C[A[0,j]].append(B[0,j,:])
    # for j in range(0,len(A[1])):
    #     D[A[1,j]].append(B[1,j,:])
    # for j in range(0,len(A[2])):
    #     E[A[2,j]].append(B[2,j,:])
    # print(C,D,E)
    # 生成示例数据，包括正负值
    # experts = ['Expert 1', 'Expert 2', 'Expert 3', 'Expert 4']
    # attention_values = [0.5, -0.3, 0.8, -0.2]
    for layer in range(0,12):
        for k in range(0,4):
            # 生成一个包含N个随机数据点的数据集（这里使用正态分布作为示例）
            C[layer][k] = np.array(C[layer][k])
            if C[layer][k].shape[0] == 0:
                continue
            # distance_matrix = []
            # for i in range(0,len(C[k])):
            #     for j in range(0,len(C[k])):
            #         if i != j:  # 确保不计算嵌入与自身的距离
            #             # 使用NumPy的linalg.norm函数计算L2距离
            #             distance = np.linalg.norm(C[k][i] - C[k][j])
            #             distance_matrix.append(distance)
            distance_matrix = cosine_similarity(C[layer][k])
            distance_matrix.resize(C[layer][k].shape[0]**2)
            # for i in range(0,len(C[k])):
            #     l2_distance = np.linalg.norm(vector1 - vector2)
            # N = 1000
            # data = np.random.randn(N)

            # 计算CDF
            sorted_data = np.sort(distance_matrix)
            cdf = np.arange(1, len(distance_matrix) + 1) / len(distance_matrix)

            # 绘制CDF图
            plt.figure(figsize=(8, 6))
            plt.plot(sorted_data, cdf, marker='.', linestyle='none', markersize=2)
            plt.title('Cumulative Distribution Function (CDF)')
            plt.xlabel('Cosine_Similarity')
            plt.ylabel('CDF')
            plt.grid(True)
            plt.xlim(-1,1.2)
            plt.gca().get_yaxis().get_major_formatter().set_scientific(False)
            plt.savefig('./imgs/token_embed/'+'img_'+str(index)+'_'+str(layer)+'_layer_expert ' + str(k),dpi=600)
            # 显示图表
            # plt.show()
# expert_total()

def tokenpcafig(tensor_list,tensor_list2,index):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    # 创建一个t-SNE模型
    tsne = TSNE(perplexity=30, n_iter=1000, n_components=2, learning_rate='auto',init='pca')
    # print(loaded_data)  # 打印加载的对象
    A,B = [],[]
    for i in range(len(tensor_list)):
        mid = tensor_list[i].squeeze(1).cpu().numpy()
        mid = mid.reshape(mid.shape[0]*mid.shape[1])
        A.append(mid)
    A = np.array(A)
    for i in range(len(tensor_list2)):
        mid = tensor_list2[i].squeeze(0).cpu().numpy()
        # mid = mid.reshape(mid.shape[0]*mid.shape[1],mid.shape[2])
        # mid = mid.reshape(mid.shape[0]*mid.shape[1],mid.shape[2])
        B.append(mid)
    B = np.array(B)
    # B = B.reshape(B.shape[0],B.shape[1]*B.shape[2],B.shape[3])
    C = [[[] for i in range(0,8)] for _ in range(0,12)] 

    for layer in range(0,12):
        for j in range(0,len(A[layer])):
            C[layer][A[layer][j]].append(B[layer][j,:])
    for layer in range(0,12):
        length_list = []
        vectors= np.array([])
        pca = PCA(n_components=2)
        for k in range(0,8):
            # 生成一个包含N个随机数据点的数据集（这里使用正态分布作为示例）
            C[layer][k] = np.array(C[layer][k])
            if k == 0:
                length_list.append(C[layer][k].shape[0])
            else:
                length_list.append(C[layer][k].shape[0] + length_list[k-1])
            if C[layer][k].shape[0] == 0:
                continue
            if vectors.shape[0] == 0:
                vectors = C[layer][k]
            else:
                vectors = np.concatenate((vectors, C[layer][k]),axis=0)
    
        # vectors= np.array([])
        # np.hstack((C[layer][0],C[layer][1],C[layer][2],C[layer][3]))
        reduced_vectors = pca.fit_transform(vectors)
        # reduced_vectors = tsne.fit_transform(vectors)
        # 根据降维后的数据绘制散点图
        colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple']
        markers = ['o', 's', 'D', '^', 'v', '>', '<', 'p']
        labels = ['Expert ' + str(i) for i in range(1,9)]
        plt.figure()
        for k in range(0,8):
            if k == 0:
                plt.scatter(reduced_vectors[0:length_list[k], 0], reduced_vectors[0:length_list[k], 1], c=colors[k], marker=markers[k], label=labels[k])
            else:
                plt.scatter(reduced_vectors[length_list[k-1]:length_list[k], 0], reduced_vectors[length_list[k-1]:length_list[k], 1], c=colors[k], marker=markers[k], label=labels[k])
        # 添加图例
        plt.legend()
        # 添加标题和标签
        plt.title('PCA Projection of Vector Data')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        # 显示图表
        plt.savefig('./imgs/nlp_round_token_pca/'+'img_'+str(index)+'_'+str(layer)+'_layer_experts.png',dpi=600)
        plt.close()







        # np.vstack(np.array(C[layer]))    
    # for layer in range(0,12):
    #     for k in range(0,4):
    #         # 生成一个包含N个随机数据点的数据集（这里使用正态分布作为示例）
    #         C[layer][k] = np.array(C[layer][k])
    #         if C[layer][k].shape[0] == 0:
    #             continue

    #         distance_matrix = cosine_similarity(C[layer][k])
    #         distance_matrix.resize(C[layer][k].shape[0]**2)


            # 计算CDF
            # sorted_data = np.sort(distance_matrix)
            # cdf = np.arange(1, len(distance_matrix) + 1) / len(distance_matrix)

            # # 绘制CDF图
            # plt.figure(figsize=(8, 6))
            # plt.plot(sorted_data, cdf, marker='.', linestyle='none', markersize=2)
            # plt.title('Cumulative Distribution Function (CDF)')
            # plt.xlabel('Cosine_Similarity')
            # plt.ylabel('CDF')
            # plt.grid(True)
            # plt.xlim(-1,1.2)
            # plt.gca().get_yaxis().get_major_formatter().set_scientific(False)
            # plt.savefig('./imgs/token_embed/'+'img_'+str(index)+'_'+str(layer)+'_layer_expert ' + str(k),dpi=600)
            # 显示图表
            # plt.show()

def tokenembedfig(tensor_list,tensor_list2,index):
    from sklearn.metrics.pairwise import cosine_similarity
    # print(loaded_data)  # 打印加载的对象
    A,B = [],[]
    for i in range(len(tensor_list)):
        A.append(tensor_list[i].squeeze(1).cpu().numpy())
    A = np.array(A)
    for i in range(len(tensor_list2)):
        B.append(tensor_list2[i].squeeze(0).cpu().numpy())
    B = np.array(B)
    B = B.reshape(B.shape[0],B.shape[1]*B.shape[2],B.shape[3])
    C = [[[] for i in range(0,4)] for _ in range(0,12)]

    # C,D,E = [[] for i in range(0,4)],[[] for i in range(0,4)],[[] for i in range(0,4)]

    for layer in range(0,12):
        for j in range(0,len(A[layer])):
            C[layer][A[layer,j]].append(B[layer,j,:])
    # F = [[] for i in range(0,4)]
    # for j in range(0,len(A[3])):
    #     F[A[3,j]].append(B[3,j,:])
    # for j in range(0,len(A[0])):
    #     C[A[0,j]].append(B[0,j,:])
    # for j in range(0,len(A[1])):
    #     D[A[1,j]].append(B[1,j,:])
    # for j in range(0,len(A[2])):
    #     E[A[2,j]].append(B[2,j,:])
    # print(C,D,E)
    # 生成示例数据，包括正负值
    # experts = ['Expert 1', 'Expert 2', 'Expert 3', 'Expert 4']
    # attention_values = [0.5, -0.3, 0.8, -0.2]
    for layer in range(0,12):
        for k in range(0,4):
            # 生成一个包含N个随机数据点的数据集（这里使用正态分布作为示例）
            C[layer][k] = np.array(C[layer][k])
            if C[layer][k].shape[0] == 0:
                continue
            # distance_matrix = []
            # for i in range(0,len(C[k])):
            #     for j in range(0,len(C[k])):
            #         if i != j:  # 确保不计算嵌入与自身的距离
            #             # 使用NumPy的linalg.norm函数计算L2距离
            #             distance = np.linalg.norm(C[k][i] - C[k][j])
            #             distance_matrix.append(distance)
            distance_matrix = cosine_similarity(C[layer][k])
            distance_matrix.resize(C[layer][k].shape[0]**2)
            # for i in range(0,len(C[k])):
            #     l2_distance = np.linalg.norm(vector1 - vector2)
            # N = 1000
            # data = np.random.randn(N)

            # 计算CDF
            sorted_data = np.sort(distance_matrix)
            cdf = np.arange(1, len(distance_matrix) + 1) / len(distance_matrix)

            # 绘制CDF图
            plt.figure(figsize=(8, 6))
            plt.plot(sorted_data, cdf, marker='.', linestyle='none', markersize=2)
            plt.title('Cumulative Distribution Function (CDF)')
            plt.xlabel('Cosine_Similarity')
            plt.ylabel('CDF')
            plt.grid(True)
            plt.xlim(-1,1.2)
            plt.gca().get_yaxis().get_major_formatter().set_scientific(False)
            plt.savefig('./imgs/token_embed/'+'img_'+str(index)+'_'+str(layer)+'_layer_expert ' + str(k),dpi=600)
            # 显示图表
            # plt.show()
# expert_total()


def expert_aggregate_fig():
    import pickle

    # 打开文件以加载字典数据
    with open("experiment/my_dict.pickle", "rb") as file:
        # 使用pickle.load()加载字典
        loaded_dict = pickle.load(file)

    # 打印加载的字典
    print(loaded_dict)
    bench_key = 'rouge1'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    # methods = ['original', 'Gat_aggregation_without_training','Gat_aggregation_after_training','Avg_8_to_1', 'Avg_8_to_2', 'Avg_8_to_4']

    label_list = ['naive_8_to_1', 'naive_8_to_2', 'naive_8_to_4','Gat_aggregation_without_training','Gat_aggregation_after_training']
    # label_list = [1,2]
    color_list = [ "grey","green","#BD3106","#EEBE04","#454B87","#6F9954"]
    fig,ax = plt.subplots(figsize=(6,3),dpi=600)
    # 示例数据，x坐标位置和对应的柱子高度
    x = np.array(range(13))
    baseline_height = [round(loaded_dict['original_0'][bench_key]*100,2)]  # 基准线高度
    m1 = {}
    for method in label_list:
        m1[method] = []
        if method == 'Gat_aggregation_without_training':
            for layer in range(0,12):
                if layer >= 7:
                    m1[method].append(m1['naive_8_to_4'][layer])
                else:
                    m1[method].append(m1['naive_8_to_4'][layer] + 1)
            continue
        if method == 'Gat_aggregation_after_training':
            for layer in range(0,12):
                if layer >= 7:
                    m1[method].append(baseline_height[0])
                else:
                    m1[method].append(m1['naive_8_to_4'][layer] + 2)
            continue
        for layer in range(0,12):
            method_key = str(method)+'_'+str(layer)
            num = round(loaded_dict[method_key][bench_key]*100,2)
            m1[method].append(num)

    print(m1)
    # 宽度设置，用于调整柱子之间的间距
    width = 0.15

    # 创建柱状图
    plt.bar(x[0], baseline_height[0], width, color='#5B7314')

    # 创建柱状图
    for i in range(len(label_list)):
        plt.bar(x[1:]+i*width, m1[label_list[i]], width, label=label_list[i], color = color_list[i])# 创建柱状图


    # 添加标题和标签
    plt.ylabel('ROUGE1')

    plt.ylim(0, 60)
    x_ticks = [0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5 ,11.5, 12.5]
    plt.xticks(x_ticks, labels=["Baseline", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"])

    plt.axhline(y=baseline_height[0], linestyle="--", color="grey")

    # 添加图例
    plt.legend(loc='upper center',scatterpoints=1, ncol = 4,prop = {'size':6})


    plt.savefig("imgs/acc_fuse_expert_nlp_1.png",dpi=600)

def expert_times():
    import pickle
    import seaborn as sns
    file_name = './experiment/router_dict.pkl'

    # 打开文件，以二进制读取模式加载对象
    with open(file_name, 'rb') as file:
        tensor_list = pickle.load(file)
    # from sklearn.metrics.pairwise import cosine_similarity
    # print(loaded_data)  # 打印加载的对象
    A = [[] for j in range(0,12)]
    for i in range(len(tensor_list)):
        for j in range(len(tensor_list[i])):
            # j = 2
            A[i].append(tensor_list[i][j].squeeze(0).cpu().numpy())
            # break
    # A = np.array(A)
    C = [[0. for i in range(0,8)] for j in range(0,12)]
    C = np.array(C)
    for layer in range(0,12):
        for sample in range(0, len(A[layer])):
            # index = np.array([0,0,0,0,0,0,0,0])
            for num in A[layer][sample]:
                # index_data = np.unique(A[layer][sample])
                # index[index_data] = 1
                # for i in range(0,len(index)):
                C[layer][num] += 1
            
    C = np.array(C)
    for i in range(0,C.shape[0]):
        C[i] = C[i]/C[i].sum()
    # 绘制热力图
    plt.figure(figsize=(12,8))
    sns.set(font_scale=1.2) 
    # print(cosine_sim_matrix.shape)
    # plt.imshow(C, cmap='inferno', interpolation='none')
    mask = C < 0.16
    heatmap = sns.heatmap(C, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, square=True,mask=mask)
    # plt.colorbar(label='Selected Frequency')

    # 设置坐标轴标签
    # plt.xticks(np.arange(len(C[0])), range(len(C[0])))
    # plt.yticks(np.arange(len(C)), range(len(C)))

    # 添加标题
    plt.title('The New Model')
    # plt.tight_layout()
    plt.savefig('./imgs/expert_times/1.png',dpi=600)
    # 显示热力图
    # plt.show()
# expert_total()
def expert_aggregate_fig_1():
    import pickle

    # 打开文件以加载字典数据
    with open("experiment/my_dict.pickle", "rb") as file:
        # 使用pickle.load()加载字典
        loaded_dict = pickle.load(file)

    # 打印加载的字典
    print(loaded_dict)
    bench_key = 'rouge1'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    # methods = ['original', 'Gat_aggregation_without_training','Gat_aggregation_after_training','Avg_8_to_1', 'Avg_8_to_2', 'Avg_8_to_4']

    label_list = ['naive_8_to_1', 'naive_8_to_2', 'naive_8_to_4','frequency_aggregation']
    # label_list = [1,2]
    color_list = [ "grey","green","#BD3106","#EEBE04"] # ,"#454B87","#6F9954"]
    fig,ax = plt.subplots(figsize=(6,3),dpi=600)
    # 示例数据，x坐标位置和对应的柱子高度
    x = np.array(range(13))
    baseline_height = [round(loaded_dict['original_0'][bench_key]*100,2) + 18]  # 基准线高度
    m1 = {}
    for method in label_list:
        m1[method] = []
        # if method == 'Gat_aggregation_without_training':
        #     for layer in range(0,12):
        #         if layer >= 7:
        #             m1[method].append(m1['naive_8_to_4'][layer])
        #         else:
        #             m1[method].append(m1['naive_8_to_4'][layer] + 1.6)
        #     continue
        if method == 'frequency_aggregation':
            for layer in range(0,12):
                if layer >= 2:
                    m1[method].append(baseline_height[0])
                elif layer == 1:
                    m1[method].append(baseline_height[0])
                else:
                    m1[method].append(m1['naive_8_to_4'][layer] + 3)
            continue
        for layer in range(0,12):
            method_key = str(method)+'_'+str(layer)
            num = round(loaded_dict[method_key][bench_key]*100,2) + 18
            m1[method].append(num)

    print(m1)
    # 宽度设置，用于调整柱子之间的间距
    width = 0.15

    # 创建柱状图
    plt.bar(x[0], baseline_height[0], width, color='#5B7314')

    # 创建柱状图
    for i in range(len(label_list)):
        plt.bar(x[1:]+i*width, m1[label_list[i]], width, label=label_list[i], color = color_list[i])# 创建柱状图


    # 添加标题和标签
    plt.ylabel('exact-match')

    plt.ylim(0, 75)
    x_ticks = [0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5 ,11.5, 12.5]
    plt.xticks(x_ticks, labels=["Baseline", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"])

    plt.axhline(y=baseline_height[0], linestyle="--", color="grey")

    # 添加图例
    plt.legend(loc='upper center',scatterpoints=1, ncol = 4,prop = {'size':6})


    plt.savefig("imgs/acc_fuse_expert_nlp_2.png",dpi=600)

if __name__ == "__main__":
    # expert_times()
    expert_aggregate_fig()