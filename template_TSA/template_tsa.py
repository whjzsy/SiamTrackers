import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


_EPSILON = 1e-6
# 实现目标注意力机制，用于更新目标特征。
# 单通道的目标特征更新。
# SiamRPN++需要对三个通道的目标特征进行更新。所以需要一个多通道的目标注意力机制。
class Template_TSA(nn.Module):
    def __init__(self, in_channels, arfa = 2 , inter_channels=None):
        super(Template_TSA, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // arfa
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv1d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)

        self.get_ht = nn.AdaptiveAvgPool2d((1, 1))
        self.get_residual = nn.Sequential(
                nn.Linear(in_features=256, out_features=256, bias=True),
                nn.Sigmoid()
                )
        self.get_strength = nn.Sequential(
                nn.Linear(in_features=256, out_features=1, bias=True),
                nn.Softplus()
        )
        nn.init.ones_(self.get_strength[0].bias)

    def _vector_norms(self, m):
        squared_norms = torch.sum(m * m, dim=2, keepdim=True)
        return torch.sqrt(squared_norms + _EPSILON)

    def _weighted_softmax(self, activations, strengths):

        sharp_activations = activations * strengths
        softmax_weights = F.softmax(sharp_activations, dim = 1)
        return softmax_weights

    def cosine_similarity(self, memory, keys, strengths):

        # Calculates the inner product between the query vector and words in memory.
        keys = keys.unsqueeze(1)
        memory_adjoint = memory.permute(0, 2, 1)
        dot = torch.matmul(keys, memory_adjoint)

        # Outer product to compute denominator (euclidean norm of query and memory).
        memory_norms = self._vector_norms(memory)
        memory_norms_adjoint = memory_norms.permute(0, 2, 1)
        key_norms = self._vector_norms(keys)
        norm = torch.matmul(key_norms, memory_norms_adjoint)

        # Calculates cosine similarity between the query vector and words in memory.
        similarity = dot / (norm + _EPSILON)

        return self._weighted_softmax(similarity.squeeze(1), strengths)

    def forward(self, init_template, memory, template):
        '''
        :param init_template: torch.size(1, 256, 7, 7)
        :param x: (batchsize, d: 7 * 7 * 256, T = 1 + t)
        :return: z: (batchsize, d: 7 * 7 * 256 , 1)
        返回下一次模板匹配的目标模板特征。
        '''
        batch_size = template.size(0)
        # 处理memory，将memory从列表转换成tensor, [batchsize, T, 256, 7, 7]
        for i in range(len(memory)):
            memory[i] = memory[i].view(batch_size, 1, 256, 7, 7)
        memory = torch.cat(memory, dim = 1)

        # -----------------【get residual vector】-----------------
        # 根据当前的模板得到 residual_vector， read_strength
        h_t = self.get_ht(template).view(batch_size, -1)
        residual_vector = self.get_residual(h_t)
        read_strength = self.get_strength(h_t)

        # -----------------【get read-weight】--------------------------
        # 简单处理一下init_template张量的格式,处理后为[batchsize, 256*7*7]
        batch_size = template.size(0)
        template = template.contiguous().view(batch_size, self.in_channels, 1)
        # 将memory从列表中的张量拼接起来，处理后的大小为[batchsize, T, 256*7*7]
        read_key = self.g(template).view(batch_size, self.inter_channels) #[]
        memorys = memory.view(batch_size, -1, self.in_channels).permute(0, 2, 1) #[batchsize, 256*7*7, T]
        memorys = self.theta(memorys).permute(0, 2, 1) #[batchsize, T, 256*7*7 // arfa]
        read_weight = self.cosine_similarity(memorys, read_key, read_strength)

        # ---------------------【read memory】---------------------------
        read_weight_expand = torch.reshape(read_weight, [-1, memory.size(1), 1, 1, 1])

        residual_vector = torch.reshape(residual_vector, [-1, 1, 256, 1, 1])
        retr_template = torch.sum(residual_vector * (read_weight_expand * memory), [1])

        z = retr_template +  init_template
        return z

class Template_MTSA(nn.Module):
    def __init__(self, in_channels, arfa = 2 , inter_channels=None):
        super(Template_MTSA, self).__init__()
        for i in range(3):
            self.add_module('template_tsa' + str(i),
                            Template_TSA(in_channels, arfa))

    def forward(self, init_templates, memorys, template):
        '''
        :param x: (batchsize, d: 15 * 15 * 112, T = 1 + t)
        :return: z: (batchsize, d: 15 * 15 * 112 , 1)
        返回下一次模板匹配的目标模板特征。
        '''
        z = []
        for idx, (init_template, memory, template_i) in enumerate(zip(init_templates, memorys, template)):
            template_tsa = getattr(self, 'template_tsa' + str(idx))
            update_template = template_tsa(init_template, memory, template_i)
            z.append(update_template)
        return z

if __name__ == '__main__':
    import torch
    x = []
    for i in range(3):
        x_t = torch.zeros(4, 256, 7, 7)
        x.append(x_t)

    T = np.random.randint(1, 16)
    memorys = [[],[],[]]
    for j in range(3):
        for i in range(T):
            memory = torch.zeros(4, 256, 7, 7)
            memorys[j].append(memory)
    z = []
    for i in range(3):
        z_t = torch.zeros(4, 256, 7, 7)
        z.append(z_t)
    net = Template_MTSA(in_channels= 256 * 7 * 7)
    out = net(x, memorys, z)
    print(len(out))
    for i in range(len(out)):
        print(out[i].shape)