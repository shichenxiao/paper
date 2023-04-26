###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data
import json
import os
import os.path
from ordered_set import OrderedSet
from transformers import BertTokenizer

bert_model = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model)
MAX_LEN = 128
def make_dataset_with_labels(self,dir):
    # assert os.path.isdir(dir), '%s is not a valid directory' % dir
    # sentences = []
    # # 临时存放每一个句子
    # sentence = []
    # all_label = []
    # label = []
    # max_len=50
    # for line in open(os.path.join(dir,'train.bmes')):
    #     # 去掉两边空格
    #     line = line.strip()
    #     # 首先判断是不是空，如果是则表示句子和句子之间的分割点
    #     if not line:
    #         if len(sentence) > 0:
    #             sentences.append(sentence)
    #             all_label.append(label)
    #             # 清空sentence表示一句话完结
    #             sentence = []
    #             label = []
    #     else:
    #         if line[0] == " ":
    #             continue
    #
    #         else:
    #             word, l = line.split()
    #             # assert len(word)>=2
    #             label.append(self.lables2id[l])
    #             sentence.append(self.words2id[word])
    #
    #
    # # 循环走完判断防止最后一个句子没有进入到句子集合中
    # if len(sentence) > 0:
    #     sentences.append(sentence)
    #     all_label.append(label)
    #
    # for data in sentences:
    #     word_len = len(data)
    #     if word_len < max_len:
    #         for i in range(max_len - word_len):
    #             data.append(self.words2id['pad'])
    #     elif word_len > max_len:
    #         # data = data[:self.p.max_len]
    #         del data[max_len:]
    #
    # for data in all_label:
    #     word_len = len(data)
    #     if word_len < max_len:
    #         for i in range(max_len - word_len):
    #             data.append(self.lables2id['O'])
    #     elif word_len > max_len:
    #         del data[max_len:]
    # return sentences,all_label

    # with open(dir, 'r', encoding='utf-8') as fr:
    #     entries = fr.read().strip().split('\n\n')
    # sents, tags_li = [], [] # list of lists
    # x, y ,sentences,all_label= [], [],[],[]
    # is_heads = []
    # for entry in entries:
    #     words = [line.split()[0] for line in entry.splitlines()]
    #     tags = ([line.split()[-1] for line in entry.splitlines()])
    #     if len(words) > MAX_LEN:
    #         # 先对句号分段
    #         word, tag = [], []
    #         for char, t in zip(words, tags):
    #
    #             if char != '。':
    #                 if char != '\ue236':  # 测试集中有这个字符
    #                     word.append(char)
    #                     tag.append(t)
    #             else:
    #                 sents.append(["[CLS]"] + word[:MAX_LEN] + ["[SEP]"])
    #                 tags_li.append(['[CLS]'] + tag[:MAX_LEN] + ['[SEP]'])
    #                 word, tag = [], []
    #                 # 最后的末尾
    #         if len(word):
    #             sents.append(["[CLS]"] + word[:MAX_LEN] + ["[SEP]"])
    #             tags_li.append(['[CLS]'] + tag[:MAX_LEN] + ['[SEP]'])
    #             word, tag = [], []
    #     else:
    #         sents.append(["[CLS]"] + words[:MAX_LEN] + ["[SEP]"])
    #         tags_li.append(['[CLS]'] + tags[:MAX_LEN] + ['[SEP]'])
    #
    # return sents, tags_li

    D = []
    source_sent = []
    sents=[]
    tags=[]
    source_domain = 0
    with open(dir, "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line.strip())
            sequences, labels = line[0], line[1]
            sents.append(sequences)
            tags.append(labels)
            source_sent.append((sequences, labels))
    for source in source_sent:
        sequences, labels = source
        D.append({'text': sequences, 'label': labels, 'source': (sequences, source_domain)})
    return sents, tags


#
def make_dataset_classwise(self,dir, category):
    # assert os.path.isdir(dir), '%s is not a valid directory' % dir
    #
    # # images = []
    # # for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
    # #     for fname in fnames:
    # #         dirname = os.path.split(root)[-1]
    # #         if dirname != category:
    # #             continue
    # #         if is_image_file(fname):
    # #             path = os.path.join(root, fname)
    # #             images.append(path)
    # #
    # # return images
    # assert os.path.isdir(dir), '%s is not a valid directory' % dir
    # sentences = []
    # # 临时存放每一个句子
    # sentence = []
    # all_label = []
    # label = []
    # max_len = 50
    # for line in open(os.path.join(dir, 'train.bmes')):
    #     # 去掉两边空格
    #     line = line.strip()
    #     # 首先判断是不是空，如果是则表示句子和句子之间的分割点
    #     # if not line:
    #     #     if len(sentence) > 0:
    #     #         sentences.append(sentence)
    #     #         all_label.append(label)
    #     #         # 清空sentence表示一句话完结
    #     #         sentence = []
    #     #         label = []
    #     if line:
    #         if line[0] == " ":
    #             continue
    #         else:
    #             word, l = line.split()
    #             # assert len(word)>=2
    #             if l!=category:
    #                 continue
    #             sentence.append(self.words2id[word])
    #
    #             label.append(self.lables2id[l])

    D = []
    source_sent = []
    sents = []
    tags = []
    source_domain = 0
    with open(dir, "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line.strip())
            sequences, labels = line[0], line[1]
            sents.append(sequences)
            tags.append(labels)
            source_sent.append((sequences, labels))
    for source in source_sent:
        sequences, labels = source
        D.append({'text': sequences, 'label': labels, 'source': (sequences, source_domain)})

    word=[]
    maxlen = 128
    totalLen = len(sents)
    for i in range(totalLen):
        sentLen=len(sents[i])
        for j in range(sentLen):
            if sentLen < maxlen:
                for m in range(maxlen-sentLen):
                    word.append(9)
            else:
                if category == tags[i][j]:
                    word.append(tokenizer.convert_tokens_to_ids(sents[i][j]))


    return word
#
def make_dataset(self,dir):
    # assert os.path.isdir(dir), '%s is not a valid directory' % dir
    # sentences = []
    # # 临时存放每一个句子
    # sentence = []
    # all_label = []
    # label = []
    # max_len = 50
    # for line in open(os.path.join(dir, 'train.bmes')):
    #     # 去掉两边空格
    #     line = line.strip()
    #     # 首先判断是不是空，如果是则表示句子和句子之间的分割点
    #     if not line:
    #         if len(sentence) > 0:
    #             sentences.append(sentence)
    #             all_label.append(label)
    #             # 清空sentence表示一句话完结
    #             sentence = []
    #             label = []
    #     else:
    #         if line[0] == " ":
    #             continue
    #         else:
    #             word, l = line.split()
    #             # assert len(word)>=2
    #             sentence.append(self.words2id[word])
    #
    #             label.append(self.lables2id[l])
    #             # sentence.append(word)
    #             #
    #             # label.append(l)
    # # 循环走完判断防止最后一个句子没有进入到句子集合中
    # if len(sentence) > 0:
    #     sentences.append(sentence)
    #     all_label.append(label)
    #
    # for data in sentences:
    #     word_len = len(data)
    #     if word_len < max_len:
    #         for i in range(max_len - word_len):
    #             data.append(self.words2id['pad'])
    #             # data.append('pad')
    #     elif word_len > max_len:
    #         # data = data[:self.p.max_len]
    #         del data[max_len:]
    #
    # for data in all_label:
    #     word_len = len(data)
    #     if word_len < max_len:
    #         for i in range(max_len - word_len):
    #             data.append(self.lables2id['O'])
    #             # data.append('O')
    #     elif word_len > max_len:
    #         del data[max_len:]
    # return sentences
    D = []
    source_sent = []
    sents = []
    tags = []
    source_domain = 0
    with open(dir, "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line.strip())
            sequences, labels = line[0], line[1]
            sents.append(sequences)
            tags.append(labels)
            source_sent.append((sequences, labels))
    for source in source_sent:
        sequences, labels = source
        D.append({'text': sequences, 'label': labels, 'source': (sequences, source_domain)})
    return sents, tags

def load_data():
    words, labels = OrderedSet(), OrderedSet()
    words.add('pad')

    splits = ['train', 'test']
    dataset = ['MSRANER', 'WeiboNER']
    for split in splits:
        for data in dataset:
            for line in open('../experiments/{}/{}.bmes'.format(data, split)):
                line = line.rstrip("\n")
                text = line.strip().split(' ')
                if (len(text) != 2):
                    continue
                words.add(text[0])
                labels.add(text[1])

    words2id = {word: idx for idx, word in enumerate(words)}
    lables2id = {label: idx for idx, label in enumerate(labels)}

    id2words = {idx: word for word, idx in words2id.items()}
    id2labels = {idx: label for label, idx in lables2id.items()}
    return words2id,lables2id,id2words,id2labels