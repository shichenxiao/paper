import torch.nn as nn
import torch
class IDCNN(nn.Module):
    def __init__(self, input_size, filters, kernel_size=3, num_block=4):
        super(IDCNN, self).__init__()
        self.layers = [
            {"dilation": 1},
            {"dilation": 1},
            {"dilation": 2}]
        net = nn.Sequential()
        norms_1 = nn.ModuleList([LayerNorm(50) for _ in range(len(self.layers))])
        norms_2 = nn.ModuleList([LayerNorm(50) for _ in range(num_block)])
        for i in range(len(self.layers)):
            dilation = self.layers[i]["dilation"]
            single_block = nn.Conv1d(in_channels=filters,
                                     out_channels=filters,
                                     kernel_size=kernel_size,
                                     dilation=dilation,
                                     padding=kernel_size // 2 + dilation - 1)
            net.add_module("layer%d"%i, single_block)
            net.add_module("relu", nn.ReLU())
            net.add_module("layernorm", norms_1[i])

        self.linear = nn.Linear(input_size, filters)
        # self.conv_init = nn.Conv1d(
        #     in_channels=input_size,
        #     out_channels=filters,
        #     kernel_size=kernel_size,
        #     padding= (kernel_size -1) // 2
        # )
        self.idcnn = nn.Sequential()


        for i in range(num_block):
            self.idcnn.add_module("block%i" % i, net)
            self.idcnn.add_module("relu", nn.ReLU())
            self.idcnn.add_module("layernorm", norms_2[i])

    def forward(self, embeddings):
        # print('input', embeddings.shape)
        # x = torch.rand(50,50,768)
        # x = x.cuda()
        # embeddings0 = self.linear(x)
        # embeddings0 = embeddings0.permute(0, 2, 1)
        # output0 = self.idcnn(embeddings0).permute(0, 2, 1)

        embeddings = self.linear(embeddings)
        embeddings = embeddings.permute(0, 2, 1)
        # embeddings = self.conv_init(embeddings)
        # print('after liner', embeddings.shape)
        output = self.idcnn(embeddings).permute(0, 2, 1)
        # print('output',output.shape)
        return output

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        if x.size(2)==1:
            std = x
        else:
            std = x.std(-1, keepdim=True)
        return self.a_2 * (x-mean) / (std + self.eps) + self.b_2



