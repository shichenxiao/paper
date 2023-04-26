import torch.utils.data
from . import single_dataset
from data.single_dataset import pad
import numpy as np
def collate_fn(sample):
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
    maxlen =60
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

    return {'sentences':words, 'word':torch.LongTensor(words2id), 'tags':tags, 'Label':torch.LongTensor(tags2id), 'seqlens':seqlens}
class TrainDatasetDataLoader(object):
    def name(self):
        return 'TrainDatasetDataLoader'

    def __init__(self, dataset_type, train, batch_size,
		dataset_root="", transform=None, classnames=None,
		paths=None, num_workers=0, labels=None,**kwargs):

        self.train = train
        self.dataset = getattr(single_dataset, dataset_type)()
        self.dataset.initialize(root=dataset_root,
                        transform=transform, classnames=classnames,
			paths=paths, labels=labels, **kwargs)

        self.classnames = classnames
        self.batch_size = batch_size

        dataset_len = len(self.dataset)
        cur_batch_size = min(dataset_len, batch_size)
        assert cur_batch_size != 0, \
            'Batch size should be nonzero value.'

        if self.train:
            drop_last = True
            sampler = torch.utils.data.RandomSampler(self.dataset)
            batch_sampler = torch.utils.data.BatchSampler(sampler,
	    			self.batch_size, drop_last)
        else:
            drop_last = False
            sampler = torch.utils.data.SequentialSampler(self.dataset)
            batch_sampler = torch.utils.data.BatchSampler(sampler,
	    			self.batch_size, drop_last)

        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                         # batch_sampler=batch_sampler,
                         batch_size=32,
                         num_workers=int(num_workers),
                         shuffle=True,
                         collate_fn=collate_fn)

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

class TrainTargetDatasetDataLoader(object):
    def name(self):
        return 'TrainTargetDatasetDataLoader'

    def __init__(self, dataset_type, train, batch_size,
		dataset_root="", transform=None, classnames=None,
		paths=None, num_workers=0, labels=None,**kwargs):

        self.train = train
        self.dataset = getattr(single_dataset, dataset_type)()
        self.dataset.initialize(root=dataset_root,
                        transform=transform, classnames=classnames,
			paths=paths, labels=labels, **kwargs)

        self.classnames = classnames
        self.batch_size = batch_size

        dataset_len = len(self.dataset)
        cur_batch_size = min(dataset_len, batch_size)
        assert cur_batch_size != 0, \
            'Batch size should be nonzero value.'

        if self.train:
            drop_last = True
            sampler = torch.utils.data.RandomSampler(self.dataset)
            batch_sampler = torch.utils.data.BatchSampler(sampler,
	    			self.batch_size, drop_last)
        else:
            drop_last = False
            sampler = torch.utils.data.SequentialSampler(self.dataset)
            batch_sampler = torch.utils.data.BatchSampler(sampler,
	    			self.batch_size, drop_last)

        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                         # batch_sampler=batch_sampler,
                         batch_size=32,
                         num_workers=int(num_workers),
                         shuffle=True,
                         collate_fn=collate_fn)

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)


