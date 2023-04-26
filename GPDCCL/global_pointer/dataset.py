import json
import torch
import random
import numpy as np
from config import args
from util import OffsetMappingProcessor
from utils.data_utils import DataSetGetter
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
id2label = {0: 'O', 1: 'PER', 2: 'ORG', 3: 'LOC', 4: 'MISC'}
label2id = {v:k for k,v in id2label.items()}
print(label2id)
domain_label2id = {'Conll2003': 0, 'SciTech': 1, 'twitter2015': 2, 'twitter2017': 3}
num_labels = len(label2id)
num_domain_labels = len(domain_label2id)

def get_entity_list(label_ids):
    entity_lists = []
    entity_list = []
    i = 0
    while i < len(label_ids) - 1:
        if label_ids[i][0] == 'B':
            entity_list.append(i)  # cls token is index 0
            start_label_type = label_ids[i].split('-')[1]
            B_idx = i
            try:
                while label_ids[i] != 'O':
                    if i > B_idx and 'B' in label_ids[i]:
                        break
                    i += 1
            except IndexError:
                pass
                # print(B_idx,i,label_ids)

            i -= 1
            entity_list.append(i)
            try:
                assert start_label_type == label_ids[i].split('-')[1]
            except:
                pass
                # print(label_ids[i],start_label_type,i, text)
                # print(label_ids)
            entity_list.append(start_label_type)
            entity_lists.append(entity_list)
            entity_list = []

        i += 1
    return entity_lists

class MyDataset(DataSetGetter):
    def load_data(self, source_file, target_file):
        D = []
        source_domain = 0
        target_domain = 1
        source_count = 0
        source_sent = []
        with open(source_file, "r", encoding="utf-8") as f:
            for line in f:
                source_count+=1
                line = json.loads(line.strip())
                sequences, labels = line[0], line[1]
                source_sent.append((sequences, labels))

        target_count = 0
        target_sent = []
        with open(target_file, "r", encoding="utf-8") as f:
            for line in f:
                target_count += 1
                line = json.loads(line.strip())
                sequences = line[0]
                target_sent.append(sequences)

        copy_times = abs(source_count) // target_count
        target_samples = []
        for _ in range(copy_times):
            target_samples.extend(target_sent)

        target_samples.extend(random.sample(target_sent, source_count - target_count*copy_times))

        assert len(target_samples) == source_count, 'number mismatch'

        for source, target in zip(source_sent, target_samples):
            sequences, labels = source
            spans = get_entity_list(labels)
            D.append({'text':sequences, 'span': spans, 'source': (sequences, source_domain), 'target':(target, target_domain)})
        return D

    def format_bert_input(self, tokens, batch_max_len):
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        special_tokens_count = 2
        if len(input_ids) > batch_max_len - special_tokens_count:
            input_ids = input_ids[:(batch_max_len - special_tokens_count)]
        input_ids += [tokenizer.sep_token_id]
        input_ids = [tokenizer.cls_token_id] + input_ids
        attention_mask = [1] * len(input_ids)

        if len(input_ids) <= batch_max_len:
            input_ids = input_ids + (batch_max_len - len(input_ids)) * [tokenizer.pad_token_id]
            attention_mask = attention_mask + (batch_max_len - len(attention_mask)) * [0]
        return input_ids, attention_mask

    def collate_fn(self, batch):
        input_ids_list, attention_mask_list, labels_list, texts_list = [], [], [], []
        batch_domain_input, batch_domain_att_mask, batch_domain_labels = [], [], []
        batch_max_len = max([len(sample['text'])+3 for sample in batch])
        batch_max_len = min(args.max_len, batch_max_len)

        for sample in batch:
            tokens = sample['text']
            spans = sample['span']
            source_sent = sample['source']
            target_sent = sample['target']
            texts_list.append(" ".join(tokens))
            input_ids, attention_mask = self.format_bert_input(tokens, batch_max_len)

            input_ids_list.append(torch.tensor(input_ids).long())
            attention_mask_list.append(torch.tensor(attention_mask) .long())
            labels = np.zeros((num_labels, batch_max_len, batch_max_len))

            if spans != []:
                for start, end, label in spans:
                    if start < end:
                        labels[label2id[label], start+1, end+1] = 1   # CLS token
            labels_list.append(torch.tensor(labels).long())

            p = random.random()
            if p<0.6:
                domain_sent = source_sent
            else:
                domain_sent = target_sent

            domain_input_ids, domain_attention_mask = self.format_bert_input(domain_sent[0], batch_max_len)
            domain_label = domain_sent[1]
            batch_domain_input.append(domain_input_ids)
            batch_domain_att_mask.append(domain_attention_mask)
            batch_domain_labels.append(domain_label)


        batch_input_ids = torch.stack(input_ids_list, dim=0)
        batch_attention_mask = torch.stack(attention_mask_list, dim=0)
        batch_labels = torch.stack(labels_list, dim=0)

        return {'input_ids': batch_input_ids, 'attention_mask': batch_attention_mask,
                'labels': batch_labels, 'domain_labels': torch.tensor(batch_domain_labels),
               'domain_attention_mask': torch.tensor( batch_domain_att_mask),'domain_input_ids': torch.tensor(batch_domain_input), 'text':texts_list}

    def decode_ent(self, text, pred_matrix, threshold=0):
        token2char_span_mapping = tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
        ent_list = {}
        for ent_type_id, token_start_index, token_end_index in zip(*np.where(pred_matrix > threshold)):
            ent_type = self.args.id2label[ent_type_id]
            ent_char_span = [token2char_span_mapping[token_start_index][0], token2char_span_mapping[token_end_index][1]]
            ent_text = text[ent_char_span[0]:ent_char_span[1]]
            ent_type_dict = ent_list.get(ent_type, {})
            ent_text_list = ent_type_dict.get(ent_text, [])
            ent_text_list.append(ent_char_span)
            ent_type_dict.update({ent_text: ent_text_list})
            ent_list.update({ent_type: ent_type_dict})
        return ent_list