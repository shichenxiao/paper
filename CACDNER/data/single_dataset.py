import os

# from distributed.protocol import torch
from ordered_set import OrderedSet
from .data_preparation import make_dataset_with_labels,make_dataset,load_data
from PIL import Image
from torch.utils.data import Dataset
# from torch.utils.data import Dataset
import torch
# class wordsAndLabel():
# from pytorch_pretrained_bert import BertTokenizer
import numpy as np
from transformers import BertTokenizer

bert_model = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model)
# VOCAB = ('<PAD>', 'O', 'I-LOC', 'B-PER', 'I-PER', 'I-ORG', 'B-LOC', 'B-ORG')
# VOCAB = ('B-LOC', 'M-LOC', 'E-LOC', 'S-LOC', 'B-PER',
#         'M-PER', 'E-PER', 'S-PER', 'B-ORG', 'M-ORG', 'E-ORG', 'S-ORG','B-GPE','M-GPE','E-GPE','S-GPE','<PAD>', '[CLS]', '[SEP]', 'O')
# VOCAB = ('B-LOC', 'M-LOC', 'E-LOC', 'S-LOC', 'B-PER',
#          'M-PER', 'E-PER', 'S-PER', 'B-ORG', 'M-ORG', 'E-ORG', 'S-ORG','<PAD>', '[CLS]', '[SEP]', 'O')
VOCAB = ('B-LOC', 'I-LOC', 'B-PER',
        'I-PER', 'B-ORG', 'I-ORG', 'B-MISC','I-MISC', 'O')
tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}
MAX_LEN = 128


class BaseDataset(Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def __getitem__(self, index):
        # # word = self.data_words[index]
        # # label = self.data_labels[index]
        # # char=[]
        # # for idx in self.data_words[index]:
        # #     char+=self.id2words.get(idx)
        # word = torch.LongTensor(self.data_words[index])
        # label = torch.LongTensor(self.data_labels[index])
        # return { 'word': word, 'Label': label}
        words, tags = self.data_words[index], self.data_labels[index]
        # x, y = [], []
        # is_heads = []
        # for w, t in zip(words, tags):
            # tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            # if tokens == []:
            #     tokens = ["[UNK]"]
        x = tokenizer.convert_tokens_to_ids(words)
        # assert len(tokens) == len(xx), f"len(tokens)={len(tokens)}, len(xx)={len(xx)}"

        # 中文没有英文wordpiece后分成几块的情况
        # is_head = [1] + [0] * (len(tokens) - 1)
        # t = [t] + ['<PAD>'] * (len(tokens) - 1)
        y = [tag2idx[tag] for tag in tags]  # (T,)

        # x.extend(xx)
        # is_heads.extend(is_head)
        #     y.extend(yy)
        assert len(x) == len(y) , f"len(x)={len(x)}, len(y)={len(y)},index = {index}"

        # seqlen
        seqlen = len(y)

        # to string
        words = " ".join(words)
        tags = " ".join(tags)
        return words, x, tags, y, seqlen

    def initialize(self, root, transform=None, **kwargs):
        self.root = root
        self.data_words = []
        self.data_labels = []
        self.transform = transform

    def __len__(self):
        return len(self.data_words)

class SingleDataset(BaseDataset):
    def initialize(self, root, classnames, transform=None, **kwargs):
        BaseDataset.initialize(self, root, transform)
        self.data_words, self.data_labels = make_dataset_with_labels(self,
				self.root)

        assert(len(self.data_words) == len(self.data_labels)), \
            'The number of images (%d) should be equal to the number of labels (%d).' % \
            (len(self.data_words), len(self.data_labels))

    def name(self):
        return 'SingleDataset'

def pad(sample):
    '''Pads to the longest sample'''
    f = lambda x: [s[x] for s in sample]
    words = f(0)
    words2id = f(1)
    # is_heads = f(2)
    tags = f(2)
    tags2id = f(3)
    seqlens = f(-1)
    # maxlen = np.array(seqlens).max()
    # f = lambda x, seqlen: [s[x] + [0] * (seqlen - len(s[x])) for s in sample]  # 0: <pad>
    maxlen =50
    word=[]
    tag=[]
    l = len(seqlens)
    for i in range(l):
        if seqlens[i]<maxlen:
            for j in range(maxlen-seqlens[i]):
                words2id[i].append(0)
                tags2id[i].append(-1)
        else:
            del words2id[i][maxlen:]
            del tags2id[i][maxlen:]
        # words2id[i]=torch.LongTensor(words2id[i])
        # tags2id[i]=torch.LongTensor(tags2id[i])

    # x = f(1, maxlen)
    # y = f(-2, maxlen)


    f = torch.LongTensor

    return {'sentences':words, 'word':torch.LongTensor(words2id),  'tags':tags, 'Label':torch.LongTensor(tags2id), 'seqlens':seqlens}

def padWithoutLabel(sample):
    '''Pads to the longest sample'''
    f = lambda x: [s[x] for s in sample]
    words = f(0)
    words2id = f(1)
    # is_heads = f(2)
    tags = f(2)
    tags2id = f(3)
    seqlens = f(-1)

    # maxlen = np.array(seqlens).max()
    # f = lambda x, seqlen: [s[x] + [0] * (seqlen - len(s[x])) for s in sample]  # 0: <pad>
    word = []
    tag = []
    maxlen = 50
    l = len(seqlens)
    for i in range(l):
        if seqlens[i] < maxlen:
            for j in range(maxlen - seqlens[i]):
                words2id[i].append(0)
                tags2id[i].append(-1)
        else:
            del words2id[i][maxlen:]
            del tags2id[i][maxlen:]

    # x = f(1, maxlen)
    # y = f(-2, maxlen)

    f = torch.LongTensor

    return {'sentences': words, 'word': torch.LongTensor(words2id),  'tags': tags,
            'seqlens': seqlens}
class BaseDatasetWithoutLabel(Dataset):
    def __init__(self):
        super(BaseDatasetWithoutLabel, self).__init__()

    def name(self):
        return 'BaseDatasetWithoutLabel'

    def __getitem__(self, index):
        # # word = self.data_words[index]
        # # label = self.data_labels[index]
        # # char = []
        # # for idx in self.data_words[index]:
        # #     char += self.id2words.get(idx)
        # datas =self.data_words[index]
        # word = torch.LongTensor(self.data_words[index])
        # return {'datas':datas,'word': word}
        # print(index)
        words, tags = self.data_words[index], self.data_labels[index]
        # x, y = [], []
        # is_heads = []
        # for w, t in zip(words, tags):
        # tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
        # if tokens == []:
        #     tokens = ["[UNK]"]
        x = tokenizer.convert_tokens_to_ids(words)
        # assert len(tokens) == len(xx), f"len(tokens)={len(tokens)}, len(xx)={len(xx)}"

        # 中文没有英文wordpiece后分成几块的情况
        # is_head = [1] + [0] * (len(tokens) - 1)
        # t = [t] + ['<PAD>'] * (len(tokens) - 1)
        y = [tag2idx[tag] for tag in tags]  # (T,)

        # x.extend(xx)
        # is_heads.extend(is_head)
        #     y.extend(yy)
        assert len(x) == len(y), f"len(x)={len(x)}, len(y)={len(y)},index = {index}"

        # seqlen
        seqlen = len(y)

        # to string
        words = " ".join(words)
        tags = " ".join(tags)
        return words, x, tags, y, seqlen

    def initialize(self, root, transform=None, **kwargs):
        self.root = root
        self.data_words = []
        self.data_labels = []
        self.transform = transform

    def __len__(self):
        return len(self.data_words)

class SingleDatasetWithoutLabel(BaseDatasetWithoutLabel):
    def initialize(self, root, transform=None, **kwargs):
        BaseDatasetWithoutLabel.initialize(self, root, transform)
        # self.words2id, self.lables2id, self.id2words, self.id2labels = load_data()
        self.data_words,self.data_labels = make_dataset(self,self.root)

    def name(self):
        return 'SingleDatasetWithoutLabel'

