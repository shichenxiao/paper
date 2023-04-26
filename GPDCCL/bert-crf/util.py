
import json
from seqeval.metrics import classification_report
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW, get_scheduler
from config import args

id2label = {0: 'O', 1: 'I-PER', 2: 'B-PER', 3: 'I-ORG', 4: 'B-LOC', 5: 'I-MISC', 6: 'B-MISC', 7: 'I-LOC', 8: 'B-ORG'}
# id2label = {0: 'O', 1: 'I-PER', 2: 'B-PER', 3: 'I-ORG', 4: 'B-ORG', 5: 'I-MISC', 6: 'B-MISC', 7: 'I-LOC', 8: 'B-LOC'}
label2id = {v:k for k,v in id2label.items()}

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
    scheduler = get_scheduler(name=args.lr_scheduler_type, optimizer=optimizer, num_warmup_steps=train_steps * args.warmup_ratios,
                                     num_training_steps=train_steps)
    return optimizer, scheduler


# def compute_metric(pred, batch):
#     # logits = torch.argmax(pred.logits, dim=2).tolist()
#     logits = torch.argmax(pred, dim=2).tolist()
#     labels = batch['labels']
#
#     out_label_list = [[] for _ in range(labels.shape[0])]
#     preds_list = [[] for _ in range(labels.shape[0])]
#     total_label = []
#     total_pred = []
#     for i in range(labels.shape[0]):  # pad token label id 对应的预测结果是不需要的
#         for j in range(labels.shape[1]):
#             if labels[i, j] != -1:
#             # if labels[i, j] != label2id['O'] and labels[i, j] != -1:
#                 out_label_list[i].append(id2label[labels[i][j].item()])
#                 preds_list[i].append(id2label[logits[i][j]])
#                 total_label.append(id2label[labels[i][j].item()])
#                 total_pred.append(id2label[logits[i][j]])
#
#     classification_report_dict = classification_report(preds_list, out_label_list, output_dict=True)
#     results = {}
#     for key0, val0 in classification_report_dict.items():
#         if key0 == 'weighted avg':
#             if isinstance(val0, dict):
#                 for key1, val1 in val0.items():
#                     if key1 == 'recall' or key1 == 'precision' or key1 == 'f1-score':
#                         results["weighted_avg_" + key1] = val1
#             else:
#                 results[key0] = val0
#     results.update({'loss':pred.loss.item()})
#     return results

def compute_metric(pred, batch):
    # logits = torch.argmax(pred.logits, dim=2).tolist()
    logits=[]
    for pre in pred:
        logits.append(torch.argmax(pre, dim=2).tolist())

    labels = batch['labels']

    out_label_list = [[] for _ in range(labels.shape[0])]
    preds_list = [[] for _ in range(labels.shape[0])]
    total_label = []
    total_pred = []
    for i in range(labels.shape[0]):  # pad token label id 对应的预测结果是不需要的
        for j in range(labels.shape[1]):
            if labels[i, j] != -1:
            # if labels[i, j] != label2id['O'] and labels[i, j] != -1:
                out_label_list[i].append(id2label[labels[i][j].item()])
                preds_list[i].append(id2label[logits[i][j]])
                total_label.append(id2label[labels[i][j].item()])
                total_pred.append(id2label[logits[i][j]])

    classification_report_dict = classification_report(preds_list, out_label_list, output_dict=True)
    results = {}
    for key0, val0 in classification_report_dict.items():
        if key0 == 'weighted avg':
            if isinstance(val0, dict):
                for key1, val1 in val0.items():
                    if key1 == 'recall' or key1 == 'precision' or key1 == 'f1-score':
                        results["weighted_avg_" + key1] = val1
            else:
                results[key0] = val0
    results.update({'loss':pred.loss.item()})
    return results
