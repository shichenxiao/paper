import os
from .data_preparation import make_dataset_with_labels, make_dataset_classwise,load_data
from PIL import Image
from torch.utils.data import Dataset
import random
from ordered_set import OrderedSet
from math import ceil
import torch
from data.single_dataset import  pad, VOCAB, tokenizer, tag2idx, idx2tag
class CategoricalDataset(Dataset):
    def __init__(self):
        super(CategoricalDataset, self).__init__()

    def initialize(self, root, classnames, class_set, 
                  batch_size, seed=None, transform=None, 
                  **kwargs):

        self.root = root
        self.transform = transform
        self.class_set = class_set
        self.load_data()
        self.data_paths = {}
        self.data_paths[self.root] = {}
        # cid = 0
        # for c in self.class_set:
        #     self.data_paths[self.root][cid] = make_dataset_classwise(self,self.root, c)
        #     cid += 1

        self.seed = seed
        self.classnames = classnames

        self.batch_sizes = {}
        self.batch_sizes[self.root] = {}
        # cid = 0
        # for c in self.class_set:
        #     batch_size = batch_size
        #     self.batch_sizes[self.root][cid] = min(batch_size, len(self.data_paths[self.root][cid]))
        #     cid += 1

    def __getitem__(self, index):
        data = {}
        root = self.root
        cur_paths = self.data_paths[root]
        
        if self.seed is not None:
            random.seed(self.seed)

        inds = random.sample(range(len(cur_paths[index])), \
                             self.batch_sizes[root][index])

        path = [cur_paths[index][ind] for ind in inds]
        data['Path'] = path
        assert(len(path) > 0)
        for p in path:
            img = Image.open(p).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)

            if not isinstance(img, torch.Tensor):
                img = torch.tensor(img)

            if 'Img' not in data:
                data['Img'] = [img]
            else:
                data['Img'] += [img]

        data['Label'] = [self.classnames.index(self.class_set[index])] * len(data['Img'])
        data['Img'] = torch.stack(data['Img'], dim=0)
        return data

    def __len__(self):
        return len(self.class_set)

    def name(self):
        return 'CategoricalDataset'

class CategoricalSTDataset(Dataset):
    def __init__(self):
        super(CategoricalSTDataset, self).__init__()

    def initialize(self,  target_labels,target_data,
                   seed=None,  **kwargs):


        self.target_labels = target_labels

        
        self.data_paths = {}
        self.target_data = target_data
        # self.data_paths['source'] = {}
        #
        # # self.words2id, self.lables2id, self.id2words, self.id2labels = load_data()
        # cid = 0
        # for c in self.class_set:
        #     self.data_paths['source'][cid] = make_dataset_classwise(self,self.source_root, c)
        #     cid += 1



        self.seed = seed

        self.batch_sizes = {}



    def __getitem__(self, index):

        words, tags = self.target_data[index], self.target_labels[index]

        return {'words':torch.LongTensor(words),'label': torch.LongTensor(tags)}

    def __len__(self):
        return len(self.target_data)

    def name(self):
        return 'CategoricalSTDataset'

