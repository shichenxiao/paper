import json
import random

import torch
import pickle as pkl
from config import args
from transformers import AutoTokenizer
from data_utils import DataSetGetter

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
id2label = {0: 'O', 1: 'I-PER', 2: 'B-PER', 3: 'I-ORG', 4: 'B-LOC', 5: 'I-MISC', 6: 'B-MISC', 7: 'I-LOC', 8: 'B-ORG'}
label2id = {v:k for k,v in id2label.items()}
print(label2id)
domain_label2id = {'Conll2003': 0, 'SciTech': 1, 'twitter2015': 2, 'twitter2017': 3}

# 继承torch Dataset父类
class MyDataset(DataSetGetter):
    # def load_data(self, source_file, target_file):
    #     D = []
    #     source_domain = 0
    #     target_domain = 1
    #     source_count = 0
    #     source_sent = []
    #     with open(source_file, "r", encoding="utf-8") as f:
    #         for line in f:
    #             source_count+=1
    #             line = json.loads(line.strip())
    #             sequences, labels = line[0], line[1]
    #             source_sent.append((sequences, labels))
    #
    #     target_count = 0
    #     target_sent = []
    #     with open(target_file, "r", encoding="utf-8") as f:
    #         for line in f:
    #             target_count += 1
    #             line = json.loads(line.strip())
    #             sequences = line[0]
    #             target_sent.append(sequences)
    #
    #     copy_times = abs(source_count) // target_count
    #     target_samples = []
    #     for _ in range(copy_times):
    #         target_samples.extend(target_sent)
    #
    #     target_samples.extend(random.sample(target_sent, source_count - target_count*copy_times))
    #
    #     assert len(target_samples) == source_count, 'number mismatch'
    #
    #     for source, target in zip(source_sent, target_samples):
    #         sequences, labels = source
    #         D.append({'text':sequences, 'label':labels, 'source': (sequences, source_domain), 'target':(target, target_domain)})
    #     return D
    def load_data(self, source_file):
        D = []
        source_sent = []
        source_domain = 0
        with open(source_file, "r", encoding="utf-8") as f:
            for line in f:
                line = json.loads(line.strip())
                sequences, labels = line[0], line[1]
                source_sent.append((sequences, labels))
        for source in source_sent:
            sequences, labels = source
            D.append({'text': sequences, 'label': labels})
        return D
    def format_bert_input(self, tokens, labels):
        label_ids = [label2id[tag] for tag in labels]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        special_tokens_count = 2
        if len(input_ids) > args.max_len - special_tokens_count:
            input_ids = input_ids[:(args.max_len - special_tokens_count)]
            label_ids = label_ids[:(args.max_len - special_tokens_count)]

        input_ids += [tokenizer.sep_token_id]
        pad_id = -1
        label_ids += [pad_id]  # [SEP] label: pad_token_label_id
        input_ids = [tokenizer.cls_token_id] + input_ids
        label_ids = [pad_id] + label_ids  # [CLS] label: pad_token_label_id

        attention_mask = [1] * len(input_ids)
        input_ids = input_ids + (args.max_len - len(input_ids)) * [tokenizer.pad_token_id]
        attention_mask = attention_mask + (args.max_len - len(attention_mask)) * [0]
        label_ids = label_ids + (args.max_len - len(label_ids)) * [pad_id]
        return input_ids, attention_mask, label_ids


    def format_bert_domain_input(self, sent):
        tokens, label = sent[0], sent[1]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        special_tokens_count = 2
        if len(input_ids) > args.max_len - special_tokens_count:
            input_ids = input_ids[:(args.max_len - special_tokens_count)]

        input_ids += [tokenizer.sep_token_id]
        pad_id = -1
        input_ids = [tokenizer.cls_token_id] + input_ids

        attention_mask = [1] * len(input_ids)
        input_ids = input_ids + (args.max_len - len(input_ids)) * [tokenizer.pad_token_id]
        attention_mask = attention_mask + (args.max_len - len(attention_mask)) * [0]

        return input_ids, attention_mask, label

    def collate_fn(self, batch):
        batch_input, batch_att_mask, batch_labels = [], [], []
        batch_domain_input, batch_domain_att_mask, batch_domain_labels = [], [], []

        for sample in batch:
            tokens = sample['text']
            labels = sample['label']
            source_sent = sample['source']
            target_sent = sample['target']
            input_ids, attention_mask, label_ids = self.format_bert_input(tokens, labels)

            batch_input.append(input_ids)
            batch_att_mask.append(attention_mask)
            batch_labels.append(label_ids)

            p = random.random()
            if p<0.6:
                domain_sent = source_sent
            else:
                domain_sent = target_sent

            domain_input_ids, domain_attention_mask, domain_label = self.format_bert_domain_input(domain_sent)
            batch_domain_input.append(domain_input_ids)
            batch_domain_att_mask.append(domain_attention_mask)
            batch_domain_labels.append(domain_label)

        return {'input_ids': torch.tensor(batch_input), 'attention_mask': torch.tensor(batch_att_mask),
                'labels': torch.tensor(batch_labels), 'domain_labels': torch.tensor(batch_domain_labels),
               'domain_attention_mask': torch.tensor( batch_domain_att_mask),'domain_input_ids': torch.tensor(batch_domain_input),
                }




