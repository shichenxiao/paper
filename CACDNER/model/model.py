import torch
import torch.nn as nn
from TorchCRF import CRF
from transformers import BertModel, BertConfig
from model.idcnn import IDCNN
from .domain_specific_module import BatchNormDomain
import torch.nn.functional as F
from . import utils as model_utils
from TorchCRF import CRF
from data.single_dataset import  pad, VOCAB, tokenizer, tag2idx, idx2tag

class FC_BN_ReLU_Domain(nn.Module):
    def __init__(self, in_dim, out_dim, num_domains_bn):
        super(FC_BN_ReLU_Domain, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = BatchNormDomain(out_dim, num_domains_bn, nn.BatchNorm1d)
        self.relu = nn.ReLU(inplace=True)
        self.bn_domain = 0

    def set_bn_domain(self, domain=0):
        assert(domain < self.num_domains_bn), "The domain id exceeds the range."
        self.bn_domain = domain
        self.bn.set_domain(self.bn_domain)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
def log_sum_exp_batch(log_Tensor, axis=-1):  # shape (batch_size,n,m)
    return torch.max(log_Tensor, axis)[0] + \
           torch.log(
               torch.exp(log_Tensor - torch.max(log_Tensor, axis)[0].view(log_Tensor.shape[0], -1, 1)).sum(axis))
class DANet(nn.Module):
    def __init__(self,num_classes, pretrained_path='../bert-base-uncased',
                  fx_pretrained=True, fc_hidden_dims=[], frozen=[],
                 num_domains_bn=2, dropout_prob=0.5):
        super(DANet, self).__init__()
        # 获取标签数
        self.tagset_size = 9
        # 获取Bert预训练模型的配置文件
        self.config = BertConfig.from_pretrained(pretrained_path)
        # 获取隐藏层维数
        # self.hidden_dim = self.config.hidden_size
        self.hidden_dim = 768
        # 加载Bert预训练模型
        self.bert = BertModel.from_pretrained(pretrained_path)
        # 定义BiLSTM层
        self.lstm = nn.LSTM(
            bidirectional=True,
            num_layers=1,
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim // 2,dropout=0.5,
            batch_first=True
        )
        # 定义dropout层
        self.dropout = nn.Dropout(dropout_prob)
        # 定义全连接层
        self.fc = nn.Linear(self.hidden_dim, self.tagset_size)
        # 定义CRF层
        self.crf = CRF(self.tagset_size)
        # self.config = BertConfig.from_pretrained(pretrained_path)
        # self.bert = BertModel.from_pretrained(pretrained_path)
        self.bn_domain = 0
        self.num_domains_bn = num_domains_bn
    def set_bn_domain(self, domain=0):
        assert (domain < self.num_domains_bn), \
            "The domain id exceeds the range."
        self.bn_domain = domain
        for m in self.modules():
            if isinstance(m, BatchNormDomain):
                m.set_domain(domain)
    def bert_enc(self, x):
        """
        通过Bert模型获取词嵌入向量
        x: [batchsize, sent_len]
        enc: [batch_size, sent_len, embedding_dim]
        """
        with torch.no_grad():
            enc = self.bert(x)[0]
            # embeddings = enc[:, 1:-1, :]
        return enc

    def get_lstm_features(self, x):
        """
        x: Bert模型中的input ids
        """
        embeds = self.bert_enc(x)
        # 过LSTM层
        lstmout, _ = self.lstm(embeds)
        # 过Dropout层
        lstmout1 = self.dropout(lstmout)
        # 过FC层
        lstm_feats = self.fc(lstmout1)

        return lstmout, lstm_feats

    def forward(self, x, y, mask=None):
        """
        前向传播函数
        """
        # 获取发射矩阵得分
        emissions = self.bert_enc(x)
        emissions = self.fc(emissions)
        # 传入CRF层
        # emissions=torch.sigmoid(emissions)
        loss = -self.crf(emissions, y, mask)
        return loss.mean()

    def predict(self, x, mask=None):
        """
        预测函数
        x：Bert模型中的input ids
        """
        emissions = self.bert_enc(x)
        emissions = self.fc(emissions)
        preds = self.crf.viterbi_decode(emissions, mask)
        # preds = [seq + [-1]*(mask.size(1)-len(seq)) for seq in preds]
        return preds

def danet(num_classes, pretrained_path,fx_pretrained=True,
          frozen=[],dropout_prob=0.5, state_dict=None,
          fc_hidden_dims=[], num_domains_bn=1, **kwargs):

    model = DANet(pretrained_path=pretrained_path,
                num_classes=num_classes, frozen=frozen,
                fx_pretrained=fx_pretrained,
                dropout_prob=dropout_prob,
                fc_hidden_dims=fc_hidden_dims,
                num_domains_bn=num_domains_bn, **kwargs)

    # if state_dict is not None:
    #     model_utils.init_weights(model, state_dict, num_domains_bn, False)

    return model

