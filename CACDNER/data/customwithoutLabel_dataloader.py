import torch.utils.data
from . import single_dataset
from data.single_dataset import padWithoutLabel,pad

class CustomWithoutLabelDatasetDataLoader(object):
    def name(self):
        return 'CustomWithoutLabelDatasetDataLoader'

    def __init__(self, dataset_type, train, batch_size,
		dataset_root="", transform=None, classnames=None,
		paths=None, labels=None,num_workers=0,**kwargs):

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
                         collate_fn=padWithoutLabel)

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)


