
from dataclasses import dataclass, field
from typing import Optional, Union, List
from transformers import TrainingArguments, HfArgumentParser
from utils.data_utils import BasicTrainingArguments



args = BasicTrainingArguments(
learning_rate=1e-5,
n_epochs=30,
train_batch_size=32,
eval_batch_size=32,
domain_adapation=True,
model_path='bert-base-uncased',
metric_key='weighted_avg_f1-score',
save_path='./experiments/',
source_data='Conll2003',
target_data="twitter2015", # twitter2017 SciTech twitter2015
data_dir="../data/process_data/",
num_labels=9,
num_domain_labels=2,
do_train=True,
do_test=True
)

args.save_path += args.target_data
if args.domain_adapation:
    args.save_path += '/domain_adapation/'
else:
    args.save_path += '/normal/'

