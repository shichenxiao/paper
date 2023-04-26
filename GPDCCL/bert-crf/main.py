import json

import torch

from utils.logger_utils import logger
from utils.trainer_utils import set_seed
from torch.utils.data import DataLoader, ConcatDataset

from dataset import MyDataset
from trainer import Trainer
from util import build_optimizer, compute_metric
from config import args
from models import BERTModel
from transformers import AutoConfig


# source_train_datasets = []
# source_dev_datasets = []
# print('source data:', args.source_data.split(' '))
# print('target data:', args.target_data)
# for source_data in args.source_data.split(' '):
#     train_dataset = MyDataset(f'{source_data}/train.json', args=args)
#     dev_dataset = MyDataset(f'{source_data}/dev.json', args=args)
#     source_train_datasets.append(train_dataset)
#     source_dev_datasets.append(dev_dataset)
#
# train_dataset = ConcatDataset(source_train_datasets)
# dev_dataset = ConcatDataset(source_dev_datasets)
# test_dataset = MyDataset(f'{args.target_data}/test.json', args=args)
#
# train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=test_dataset.collate_fn, shuffle=True)
# dev_loader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, collate_fn=test_dataset.collate_fn, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, collate_fn=test_dataset.collate_fn, shuffle=False)

train_dataset = MyDataset(f'{args.source_data}/train.json', f'{args.target_data}/train.json', args=args)
dev_dataset = MyDataset(f'{args.source_data}/dev.json', f'{args.source_data}/dev.json', args=args)
# dev_dataset = MyDataset(f'{args.source_data}/dev.json', args=args)
test_dataset = MyDataset(f'{args.target_data}/test.json', f'{args.target_data}/test.json', args=args)
# test_dataset = MyDataset(f'{args.target_data}/test.json', args=args)
train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=test_dataset.collate_fn, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, collate_fn=test_dataset.collate_fn, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, collate_fn=test_dataset.collate_fn, shuffle=False)
# model
logger.info(f'Loading model and optimizer...')
set_seed(args)



model = BERTModel.from_pretrained(args.model_path)

train_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.n_epochs

# optimizer
optimizer, scheduler = build_optimizer(model, train_steps, args)
trainer = Trainer(model,
                  args,
                  train_loader,
                  optimizer=optimizer,
                  scheduler=scheduler,
                  dev_data=dev_loader,
                  test_data=test_loader,
                  intervals=100,
                  metrics=compute_metric,
                  )
if args.do_train:
    trainer.train()


if args.do_test:
    result = trainer.test('best_bert_base_BERTModel_weighted_avg_f1-score')
