import torch
import numpy as np
from transformers import AdamW, get_scheduler
import transformers
transformers.logging.set_verbosity_error()


def build_optimizer(model, train_steps, args):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    # adam = bnb.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8, optim_bits=8)
    scheduler = get_scheduler(name=args.lr_scheduler_type, optimizer=optimizer, num_warmup_steps=train_steps * args.warmup_ratios,
                                     num_training_steps=train_steps)
    return optimizer, scheduler


class OffsetMappingProcessor(object):
    def __init__(self, tokenizer, add_special_tokens=True):
        super(OffsetMappingProcessor, self).__init__()
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens

    def get_ent2token_spans(self, text, entity_list):
        """实体列表转为token_spans
        Args:
            text (str): 原始文本
            entity_list (list): [(start, end, ent_type),(start, end, ent_type)...]
        """
        ent2token_spans = []

        inputs = self.tokenizer(text, add_special_tokens=self.add_special_tokens, return_offsets_mapping=True)
        token2char_span_mapping = inputs["offset_mapping"]
        text2tokens = self.tokenizer.tokenize(text, add_special_tokens=self.add_special_tokens)

        for ent_span in entity_list:
            ent = text[ent_span[0]:ent_span[1] + 1]
            ent2token = self.tokenizer.tokenize(ent, add_special_tokens=False)

            # 寻找ent的token_span
            token_start_indexs = [i for i, v in enumerate(text2tokens) if v == ent2token[0]]
            token_end_indexs = [i for i, v in enumerate(text2tokens) if v == ent2token[-1]]

            token_start_index = list(filter(lambda x: token2char_span_mapping[x][0] == ent_span[0], token_start_indexs))
            token_end_index = list(filter(lambda x: token2char_span_mapping[x][-1] - 1 == ent_span[1], token_end_indexs))  # token2char_span_mapping[x][-1]-1 减1是因为原始的char_span是闭区间，而token2char_span是开区间

            if len(token_start_index) == 0 or len(token_end_index) == 0:
                # print(f'[{ent}] 无法对应到 [{text}] 的token_span，已丢弃')
                continue
            token_span = (token_start_index[0], token_end_index[0], ent_span[2])
            ent2token_spans.append(token_span)

        return ent2token_spans


def kl_div_table(table_scores_stu, table_scores_teach):
    # table_scores: [bsz, seq_len, seq_len], 假设已经经过了sigmoid激活函数

    table_scores_stu = torch.clamp(table_scores_stu, min=1e-7, max=1-1e-7)
    table_scores_stu_1 = 1 - table_scores_stu
    table_score_dist_stu = torch.cat(
        [table_scores_stu.unsqueeze(-1), table_scores_stu_1.unsqueeze(-1)], dim=-1
    )

    table_scores_teach = torch.clamp(table_scores_teach, min=1e-7, max=1-1e-7)
    table_scores_teach_1 = 1 - table_scores_teach
    table_score_dist_teach = torch.cat(
        [table_scores_teach.unsqueeze(-1), table_scores_teach_1.unsqueeze(-1)], dim=-1
    )

    """Kullback-Leibler divergence D(P || Q) for discrete distributions"""
    p = torch.log(table_score_dist_stu.view(-1, 2))
    q = torch.log(table_score_dist_teach.view(-1, 2))
    # kl_score = kl_divergence(table_score_dist_stu, table_score_dist_teach, log_prob=True)

    scores = torch.sum(p.exp() * (p - q), axis=-1)
    return scores.mean()


# if self.args.rdrop and epoch > 3:
#     batch_ = batch.copy()
#     outputs_ = self._data_forward(self.model, batch_)
#     loss_ = outputs_.loss
#     kl_loss = kl_div_table(F.sigmoid(outputs.logits), F.sigmoid(outputs_.logits)) + kl_div_table(
#         F.sigmoid(outputs_.logits), F.sigmoid(outputs.logits))
#     loss = kl_loss * self.args.rdrop_alpha + (loss + loss_) / 2


def compute_metric(pred, batch):
    y_pred = pred.logits.data.cpu().numpy()
    labels = batch['labels']
    y_true = labels.data.cpu().numpy()
    pred = []
    true = []
    for b, l, start, end in zip(*np.where(y_pred > 0)):
        pred.append((b, l, start, end))
    for b, l, start, end in zip(*np.where(y_true > 0)):
        true.append((b, l, start, end))

    R = set(pred)
    T = set(true)
    X = len(R & T)
    Y = len(R)
    Z = len(T)
    precision, recall = (X + 1e-10) / (Y + 1e-10), (X + 1e-10) / (Z + 1e-10)
    f1 = precision * recall * 2 / (precision + recall)
    return {'f1': f1, 'precision': precision, 'recall': recall}

