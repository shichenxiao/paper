
import torch
import json
import numpy as np
from utils.logger_utils import logger
from utils.trainer_utils import set_seed
from torch.utils.data import DataLoader, ConcatDataset
from dataset import MyDataset
from models import GlobalPointerModel
from config import args
from trainer import Trainer
from util import build_optimizer, compute_metric


train_dataset = MyDataset(f'{args.source_data}/train.json', f'{args.target_data}/train.json', args=args)
dev_dataset = MyDataset(f'{args.source_data}/dev.json', f'{args.target_data}/dev.json', args=args)
test_dataset = MyDataset(f'{args.target_data}/test.json', f'{args.target_data}/test.json', args=args)
train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=test_dataset.collate_fn, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, collate_fn=test_dataset.collate_fn, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, collate_fn=test_dataset.collate_fn, shuffle=False)

# model
logger.info(f'Loading model and optimizer...')
set_seed(args)

model = GlobalPointerModel.from_pretrained(args.model_path)


# optimizer
train_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.n_epochs
optimizer, scheduler = build_optimizer(model, train_steps, args)

trainer = Trainer(model, args, train_loader,
                  optimizer=optimizer, scheduler=scheduler,
                  dev_data=dev_loader, test_data=dev_loader,
                  intervals=100, metrics=compute_metric)

#trainer.train()


if args.do_predict:
    trainer.test('best_bert_base_GlobalPointerModel_f1')


