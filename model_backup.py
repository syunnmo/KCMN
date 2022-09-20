import torch
import torch.nn as nn
from memory import DKVMN
import numpy as np
import utils as utils
import embedd_loss as embedd_loss
import torch.nn.functional as F
class MODEL(nn.Module):

    def __init__(self, n_exercise, batch_size, exercise_embed_dim,
                 memory_size, memory_key_state_dim, memory_value_state_dim, final_fc_dim, params,student_num=None):
        super(MODEL, self).__init__()
        self.n_exercise = n_exercise
        self.n_kc = params.n_knowledge_concept
        self.batch_size = batch_size
        self.exercise_embed_dim = exercise_embed_dim
        self.memory_size = memory_size
        self.memory_key_state_dim = memory_key_state_dim
        self.memory_value_state_dim = memory_value_state_dim
        self.final_fc_dim = final_fc_dim
        self.student_num = student_num
        self.nheads = params.num_heads
        self.alpha = params.alpha
        self.dropout = params.dropout
        self.params = params
        self.mode = params.mode

        self.read_embed_linear = nn.Linear(self.memory_value_state_dim + self.final_fc_dim, self.final_fc_dim, bias=True)
        self.predict_linear = nn.Linear(self.final_fc_dim, 1, bias=True)

        self.init_memory_key = nn.Parameter(torch.randn(self.memory_size, self.memory_key_state_dim))
        nn.init.kaiming_normal(self.init_memory_key)

        self.init_memory_value = nn.Parameter(torch.randn(self.memory_size, self.memory_value_state_dim))
        nn.init.kaiming_normal(self.init_memory_value)

        self.mem = DKVMN(memory_size=self.memory_size,
                   memory_key_state_dim=self.memory_key_state_dim,
                   memory_value_state_dim=self.memory_value_state_dim, init_memory_key=self.init_memory_key)

        memory_value = nn.Parameter(torch.cat([self.init_memory_value.unsqueeze(0) for _ in range(batch_size)], 0).data)
        self.mem.init_value_memory(memory_value)

        self.exercise_embed = nn.Embedding(self.n_exercise + 1, self.exercise_embed_dim, padding_idx=0)

        # 知识点对习题的attention
        self.exercise_kc_attentions = [
            Exercise_KC_GraphAttentionLayer(self.exercise_embed_dim, self.exercise_embed_dim, dropout=self.dropout, alpha=self.alpha, mode=self.mode,
                                            concat=True) for _ in range(self.nheads)]
        for i, attention in enumerate(self.exercise_kc_attentions):
            self.add_module('exercise_kc_attention_{}'.format(i), attention)

        self.init_params()
        self.init_embeddings()

    def init_params(self):
        nn.init.kaiming_normal(self.predict_linear.weight)
        nn.init.kaiming_normal(self.read_embed_linear.weight)
        nn.init.constant(self.read_embed_linear.bias, 0)
        nn.init.constant(self.predict_linear.bias, 0)

    def init_embeddings(self):
        nn.init.kaiming_normal(self.exercise_embed.weight)

    def forward(self, adj_exercise_kc, kc_data, exercise_data, exercise_respond_data, target, student_id=None):

        batch_size = exercise_data.shape[0]
        seqlen = exercise_data.shape[1]

        kc_node_mebedding = self.init_memory_key

        exercise_node = utils.varible(torch.linspace(1, self.n_exercise, steps=self.n_exercise).long(), self.params.gpu)
        exercise_node_embedding = self.exercise_embed(exercise_node)

        # exercise_embedding_list = [att(exercise_node_embedding, kc_node_mebedding, adj_exercise_kc) for att in self.exercise_kc_attentions]
        # exercise_embedding = utils.varible(torch.zeros(self.n_exercise, self.exercise_embed_dim), self.params.gpu)
        # for i, propagated_exercise_embedding in enumerate(exercise_embedding_list):
        #     exercise_embedding = exercise_embedding.add_(propagated_exercise_embedding)
        # exercise_embedding = exercise_embedding / (i+1)
        #
        # exercise_embedding_add_zero = torch.cat([utils.varible(torch.zeros(1, exercise_embedding.shape[1]),self.params.gpu), exercise_embedding], dim=0)

        exercise_embedding = torch.cat([att(exercise_node_embedding, kc_node_mebedding, adj_exercise_kc) for att in self.exercise_kc_attentions], dim=1).view(self.n_exercise, self.exercise_embed_dim ,self.nheads).mean(2)
        exercise_embedding_add_zero = torch.cat(
            [utils.varible(torch.zeros(1, exercise_embedding.shape[1]), self.params.gpu), exercise_embedding], dim=0)


        memory_value = nn.Parameter(torch.cat([self.init_memory_value.unsqueeze(0) for _ in range(batch_size)], 0).data)
        self.mem.init_value_memory(memory_value)

        slice_exercise_data = torch.chunk(exercise_data, seqlen, 1)
        slice_exercise_embedd_data = []
        for i, single_slice_exercise_data_index in enumerate(slice_exercise_data):
            single_slice_exercise_embedd_data = torch.index_select(exercise_embedding_add_zero, 0,
                                                                   single_slice_exercise_data_index.squeeze(1))
            slice_exercise_embedd_data.append(single_slice_exercise_embedd_data)


        slice_exercise_respond_data = torch.chunk(exercise_respond_data, seqlen, 1)
        # 全零向量拼接，答对拼右边，答错拼左边
        zeros = torch.zeros_like(exercise_embedding)
        cat1 = torch.cat((zeros, exercise_embedding), -1)
        cat2 = torch.cat((exercise_embedding, zeros), -1)
        response_embedding = torch.cat((cat1, cat2), -2)
        response_embedding_add_zero = torch.cat([utils.varible(torch.zeros(1, response_embedding.shape[1]), self.params.gpu), response_embedding],
                                                dim=0)
        slice_respond_embedd_data = []
        for i, single_slice_respond_data_index in enumerate(slice_exercise_respond_data):
            single_slice_respond_embedd_data = torch.index_select(response_embedding_add_zero, 0, single_slice_respond_data_index.squeeze(1))
            slice_respond_embedd_data.append(single_slice_respond_embedd_data)

        value_read_content_l = []
        input_embed_l = []
        for i in range(seqlen):
            ## Attention
            exercise = slice_exercise_embedd_data[i].squeeze(1)
            correlation_weight = self.mem.attention(exercise)
            if_memory_write = slice_exercise_data[i].squeeze(1).ge(1)
            if_memory_write = utils.varible(torch.FloatTensor(if_memory_write.data.tolist()), self.params.gpu)

            ## Read Process
            read_content = self.mem.read(correlation_weight)
            value_read_content_l.append(read_content)
            input_embed_l.append(exercise)
            ## Write Process
            exercise_respond = slice_respond_embedd_data[i].squeeze(1)
            new_memory_value = self.mem.write(correlation_weight, exercise_respond, if_memory_write)


        all_read_value_content = torch.cat([value_read_content_l[i].unsqueeze(1) for i in range(seqlen)], 1)
        input_embed_content = torch.cat([input_embed_l[i].unsqueeze(1) for i in range(seqlen)], 1)

        predict_input = torch.cat([all_read_value_content, input_embed_content], 2)
        read_content_embed = torch.tanh(self.read_embed_linear(predict_input.view(batch_size*seqlen, -1)))

        pred = self.predict_linear(read_content_embed)
        # predicts = torch.cat([predict_logs[i] for i in range(seqlen)], 1)
        target_1d = target                   # [batch_size * seq_len, 1]
        mask = target_1d.ge(0)               # [batch_size * seq_len, 1]
        # pred_1d = predicts.view(-1, 1)           # [batch_size * seq_len, 1]
        pred_1d = pred.view(-1, 1)           # [batch_size * seq_len, 1]

        filtered_pred = torch.masked_select(pred_1d, mask)
        filtered_target = torch.masked_select(target_1d, mask)
        predict_loss = torch.nn.functional.binary_cross_entropy_with_logits(filtered_pred, filtered_target)

        kc_exercises_embedd_loss = embedd_loss.kc_exercises_embedd_loss(adj_exercise_kc, kc_node_mebedding, exercise_embedding)

        loss = predict_loss + kc_exercises_embedd_loss

        return loss, torch.sigmoid(filtered_pred), filtered_target

class Exercise_KC_GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, mode, concat=True):
        super(Exercise_KC_GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.mode =mode

        self.W1 = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.kaiming_normal(self.W1)

        if mode == 1 or mode == 3:
            self.reduceDim = nn.Linear(in_features * 2, self.out_features, bias=True)
            nn.init.kaiming_normal(self.reduceDim.weight)
            nn.init.constant(self.reduceDim.bias, 0)
        if mode != 1:
            self.E = nn.Parameter(torch.empty(size=(in_features, out_features)))
            nn.init.kaiming_normal(self.E)
        if mode ==4:
            self.U = nn.Parameter(torch.empty(size=(in_features, 1)))
            nn.init.kaiming_normal(self.U)

        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.kaiming_normal(self.a)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, exercise_h, kc_h, adj_exercise_kc):
        if self.concat:
            kc_Wh = torch.mm(kc_h, self.W1)
            exercise_Wh = torch.mm(exercise_h, self.W1)
            a_input = self._prepare_attentional_mechanism_input(kc_Wh, exercise_Wh)
            e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj_exercise_kc > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)
            new_kc_embed = torch.matmul(attention, kc_Wh)#得到了相对每个习题的知识点信息

            if self.mode == 1:
                exercises_embedd = torch.cat((new_kc_embed, exercise_h), dim=1)
                exercises_embedd = self.reduceDim(exercises_embedd)
            if self.mode == 2:
                exercise_Eh = torch.mm(exercise_h, self.E)
                exercises_embedd = new_kc_embed.mul(exercise_Eh)
            if self.mode == 3:
                exercise_Eh = torch.mm(exercise_h, self.E)
                exercises_embedd = torch.cat((new_kc_embed, new_kc_embed.mul(exercise_Eh)), dim=1)
                exercises_embedd = self.reduceDim(exercises_embedd)
            if self.mode == 4:
                u = torch.mm(exercise_h, self.U)
                d_kt = torch.mm(new_kc_embed, self.E)
                exercises_embedd = new_kc_embed + u * d_kt
            return F.elu(exercises_embedd)


    def _prepare_attentional_mechanism_input(self, kc_Wh, exercise_Wh):
        N_kc = kc_Wh.size()[0]
        N_exercise = exercise_Wh.size()[0]
        Wh_repeated_in_chunks = exercise_Wh.repeat_interleave(N_kc, dim=0)
        Wh_repeated_alternating = kc_Wh.repeat(N_exercise, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # 返回的结果是N_exercise*N_kc*2 * self.out_features,前N行即第一个节点和其他所有节点的拼接（包括本节点的拼接）
        return all_combinations_matrix.view(N_exercise, N_kc, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



# 聚集知识点信息获得属于习题节点的知识点信息，并对差异性信息进行组合处理
class Exercise_KC_GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(Exercise_KC_GraphConvolution, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W1 = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.kaiming_normal(self.W1)
        # nn.init.xavier_uniform_(self.W1.data, gain=1.414)

        # self.W2 = nn.Linear(in_features * 2, self.out_features,bias=True)
        # nn.init.kaiming_normal(self.W2.weight)
        # nn.init.constant(self.W2.bias, 0)
        self.W2 = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.kaiming_normal(self.W2)

        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.kaiming_normal(self.a)
        # nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, exercise_h, kc_h, adj_exercise_kc):
        # 前面的nhead是聚合邻居信息
        if self.concat:
            kc_Wh = torch.mm(kc_h, self.W1)
            exercise_Wh = torch.mm(exercise_h, self.W1)
            new_kc_embed = torch.spmm(adj_exercise_kc.to(torch.float32), kc_Wh)
            # exercises_embedd = torch.cat((new_kc_embed, exercise_Wh), dim=1)
            # exercises_embedd = self.W2(exercises_embedd)
            exercises_embedd = new_kc_embed.mul(torch.mm(exercise_Wh, self.W2))
            return F.elu(exercises_embedd)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'