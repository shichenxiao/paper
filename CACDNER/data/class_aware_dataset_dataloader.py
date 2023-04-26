import torch.utils.data
from .categorical_dataset import CategoricalSTDataset
from math import ceil as ceil

def collate_fn(data):

    word = torch.stack([_['words'] for _ in data], dim=0)
    label = torch.stack([_['label'] for _ in data], dim=0)

    return {'words': word, 'label': label}

class ClassAwareDataLoader(object):
    def name(self):
        return 'ClassAwareDataLoader'

    def __init__(self,
                 target_labels=[], target_data=[],
               num_workers=0, drop_last=True,
                sampler='RandomSampler', **kwargs):
        
        # dataset type
        self.dataset = CategoricalSTDataset()

        # dataset parameters
        self.target_labels = target_labels
        self.target_data = target_data

        # loader parameters
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.sampler = sampler
        self.kwargs = kwargs
        # self.batch_size = batch_size

    def construct(self):
        self.dataset.initialize(
                target_labels=self.target_labels,
                target_data = self.target_data,
                  **self.kwargs)

        drop_last = self.drop_last
        sampler = getattr(torch.utils.data, self.sampler)(self.dataset)
        # batch_sampler = torch.utils.data.BatchSampler(sampler, self.batch_size,drop_last)

        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                         # batch_sampler=batch_sampler,
                         batch_size=32,
                         shuffle=True,
                         collate_fn=collate_fn,
                         num_workers=int(self.num_workers))

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.target_data)

