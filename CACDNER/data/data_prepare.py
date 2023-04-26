

class InputFeature(object):
    def __init__(self, input_id, label_id, input_mask):
        self.input_id = input_id
        self.label_id = label_id
        self.input_mask = input_mask

def load_vocab(vocab_file):
    '''construct word2id or label2id'''
    vocab = {}
    index = 0
    with open(vocab_file, 'r', encoding='utf-8') as fp:
        while True:
            token = fp.readline()
            if not token:
                break
            token = token.strip()  # 删除空白符
            vocab[token] = index
            index += 1
    return vocab

def read_corpus(path, max_length, label_dic, vocab):
    '''
    :param path: 数据集文件路径
    :param max_length: 句子最大长度
    :param label_dic: 标签字典
    :param vocab:
    :return:
    '''
    with open(path, 'r', encoding='utf-8') as fp:
        result = []
        words = []
        labels = []
        for line in fp:
            contends = line.strip()
            tokens = contends.split(' ')
            if len(tokens) == 2:
                words.append(tokens[0])
                labels.append(tokens[1])
            else:
                if len(contends) == 0 and len(words) > 0:
                    if len(words) > max_length - 2:
                        words = words[0:(max_length-2)]
                        labels = labels[0:(max_length-2)]
                    words = ['[CLS]'] + words + ['[SEP]']
                    labels = ['<START>'] + labels + ['<EOS>']
                    input_ids = [int(vocab[i]) if i in vocab else int(vocab['[UNK]']) for i in words]
                    label_ids = [label_dic[i] for i in labels]
                    input_mask = [1] * len(input_ids)
                    # 填充
                    if len(input_ids) < max_length:
                        input_ids.extend([0]*(max_length-len(input_ids)))
                        label_ids.extend([0]*(max_length-len(label_ids)))
                        input_mask.extend([0]*(max_length-len(input_mask)))
                    assert len(input_ids) == max_length
                    assert len(label_ids) == max_length
                    assert len(input_mask) == max_length
                    feature = InputFeature(input_id=input_ids, label_id=label_ids, input_mask=input_mask)
                    result.append(feature)
                    # 还原words、labels = []
                    words = []
                    labels = []
        return result
