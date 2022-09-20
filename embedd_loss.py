import torch
# 计算习题向量和知识点向量之间的相似度
def kc_exercises_embedd_loss(adj_exercise_kc, kc_node_mebedding, exercise_embedding):
    exercise_kc_similarity = torch.matmul(exercise_embedding, kc_node_mebedding.t())
    exercise_kc_similarity = torch.sigmoid(exercise_kc_similarity)
    loss_exercise_connected_kc = 1 - exercise_kc_similarity
    loss_exercise_disconnected_kc = exercise_kc_similarity
    zero_vec = torch.zeros_like(exercise_kc_similarity)  # 获得一个和e一样的全零
    loss_exercise_connected_kc = torch.where(adj_exercise_kc > 0, loss_exercise_connected_kc, zero_vec)
    loss_exercise_disconnected_kc = torch.where(adj_exercise_kc <= 0, loss_exercise_disconnected_kc, zero_vec)
    embedd_loss = loss_exercise_connected_kc + loss_exercise_disconnected_kc
    return embedd_loss.mean()

# 计算习题向量之间的相似度,共用相同知识点的习题应该具有一定的相似性
def exercises_embedd_loss(adj_exercise_kc, kc_node_mebedding, exercise_embedding):
    exercises_similarity = torch.matmul(exercise_embedding, exercise_embedding.t())
    exercises_similarity = torch.sigmoid(exercises_similarity)
    loss_exercises_share_kc = 1 - exercises_similarity
    loss_exercise_noshare_kc = exercises_similarity
    adj_exercises =  torch.matmul(adj_exercise_kc, adj_exercise_kc.t()).to(torch.float32)#存在共享知识点的情况时，当前位置大于1
    zero_vec = torch.zeros_like(adj_exercises)
    loss_exercises_share_kc = torch.where(adj_exercises > 0, loss_exercises_share_kc, zero_vec)
    loss_exercise_noshare_kc = torch.where(adj_exercises <= 0, loss_exercise_noshare_kc, zero_vec)
    embedd_loss = loss_exercises_share_kc + loss_exercise_noshare_kc
    return embedd_loss.mean()

# 利用欧氏距离计算embedding的损失
def kc_exercises_embedd_loss_Euclidean_Distance(self, adj_exercise_kc, kc_node_mebedding, exercise_embedding):
    euclidean_distance = self.euclidean_dist(exercise_embedding, kc_node_mebedding) #n_exercise * n_kc
        # 相似性越高，则值就越大。例如：两个相同的向量得到的euclidean_distance值为0，因此exercise_kc_similarity为1
    exercise_kc_similarity = torch.div(1, 1 + euclidean_distance)
    loss_exercise_connected_kc = 1 - exercise_kc_similarity
    loss_exercise_disconnected_kc = exercise_kc_similarity
    zero_vec = torch.zeros_like(exercise_kc_similarity)
    loss_exercise_connected_kc = torch.where(adj_exercise_kc > 0, loss_exercise_connected_kc, zero_vec)
    loss_exercise_disconnected_kc = torch.where(adj_exercise_kc <= 0, loss_exercise_disconnected_kc, zero_vec)
    embedd_loss = loss_exercise_connected_kc + loss_exercise_disconnected_kc
    # embedd_loss = loss_exercise_connected_kc.mean()
    return embedd_loss.mean()

    # 计算习题向量之间的相似度,共用相同知识点的习题应该具有一定的相似性
def exercises_embedd_loss_Euclidean_Distance(adj_exercise_kc, kc_node_mebedding, exercise_embedding):
    n_exercise, n_kc = adj_exercise_kc.shape
    euclidean_distance = euclidean_dist(exercise_embedding, exercise_embedding)
    exercises_similarity = torch.div(1, 1 + euclidean_distance)
    neg1 = torch.sum(adj_exercise_kc, dim=1, keepdim=True).expand(n_exercise, n_exercise)
    neg2 = torch.sum(adj_exercise_kc, dim=1, keepdim=True).t().expand(n_exercise, n_exercise)
    sumNeg = neg1 + neg2
    commonNeg = torch.matmul(adj_exercise_kc, adj_exercise_kc.t())
    Similarity_weight = torch.div(2 * commonNeg, sumNeg)  # 这里的0.6是超参数
    zero_vec = torch.zeros_like(commonNeg)
    loss_exercises = 0.6 * Similarity_weight - exercises_similarity
    loss_exercises = F.relu(torch.where(commonNeg > 0, loss_exercises, zero_vec))  # 保留具有共享知识点的函数
        # mask = 1 - torch.eye(n_exercise)
    return loss_exercises.mean()

def euclidean_dist(x, y):
    """
            Args:
              x: pytorch Variable, with shape [m, d]
              y: pytorch Variable, with shape [n, d]
            Returns:
              dist: pytorch Variable, with shape [m, n]
            """

    m, n = x.size(0), y.size(0)
    # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    # yy会在最后进行转置的操作
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT
    dist.addmm_(1, -2, x, y.t())
    # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def cosine_similarity3(a, b):
    '''
    a shape: [num_item_1, embedding_dim]
    b shape: [num_item_2, embedding_dim]
    return sim_matrix: [num_item_1, num_item_2]
    '''
    a = a / torch.clamp(a.norm(dim=-1, keepdim=True, p=2), min=1e-5)
    b = b / torch.clamp(b.norm(dim=-1, keepdim=True, p=2), min=1e-5)
    if len(a.shape) == 3: # 矩阵是三维
        sim = torch.bmm(a, torch.transpose(b, 1, 2))
    else:
        sim = torch.mm(a, b.t())