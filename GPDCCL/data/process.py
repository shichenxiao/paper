import json
import random

from transformers import AutoTokenizer

total_labels = ['O']
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
D = []

with open('raw_data/SciTech/tech_test') as f:
    sequences = []
    labels = []
    for line in f.readlines():
        if line != '\n':
            try:
                words, label = line.strip().split(' ')
            except:
                continue
            tokenized_words = tokenizer.tokenize(words)  # 对每个word tokenize
            for i, word in enumerate(tokenized_words):  # 如果一个word被拆分成了多个token
                sequences.append(word)
                if label == 'O':
                    labels.append(label)
                    total_labels.append(label)
                elif 'S' == label[0]:
                    if i == 0:
                        lab = f"B-{label.replace('S-', '')}"
                        labels.append(lab)
                        total_labels.append(lab)
                    else:
                        lab = f"I-{label.replace('S-', '')}"
                        labels.append(lab)

                        total_labels.append(lab)
                elif 'B' == label[0]:  # subword第一个为B，其余的为I
                    if i == 0:
                        lab = f"B-{label.replace('B-', '')}"
                        labels.append(lab)
                        total_labels.append(lab)
                    else:
                        lab = f"I-{label.replace('B-', '')}"
                        labels.append(lab)
                        total_labels.append(lab)
                elif 'I' == label[0]:
                    labels.append(label)
                    total_labels.append(label)
                elif 'E' == label[0]:
                    lab = f"I-{label.replace('E-', '')}"
                    labels.append(lab)
                    total_labels.append(lab)
                else:

                    raise TypeError('not supported label type!')
        else:
            assert len(labels) == len(sequences)
            D.append((sequences, labels))
            sequences = []
            labels = []


# with open('raw_data/twitter2017/train.txt', encoding='utf-8') as f:
#     sequences = []
#     labels = []
#     for line in f.readlines():
#         #print(line)
#         if line != '\n':
#             try:
#                 words, label = line.strip().split('\t')
#             except:
#                 continue
#             tokenized_words = tokenizer.tokenize(words)  # 对每个word tokenize
#             for i, word in enumerate(tokenized_words):  # 如果一个word被拆分成了多个token
#                 sequences.append(word)
#                 if 'OTHER' in label:
#                     label = label.replace('OTHER', 'MISC')
#                 if label == 'O':
#                     labels.append(label)
#                     total_labels.append(label)
#                 elif 'S' == label[0]:
#                     if i == 0:
#                         lab = f"B-{label.replace('S-', '')}"
#                         labels.append(lab)
#                         total_labels.append(lab)
#                     else:
#                         lab = f"I-{label.replace('S-', '')}"
#                         labels.append(lab)
#                         total_labels.append(lab)
#                 elif 'B' == label[0]:  # subword第一个为B，其余的为I
#                     if i == 0:
#                         lab = f"B-{label.replace('B-', '')}"
#                         labels.append(lab)
#                         total_labels.append(lab)
#                     else:
#                         lab = f"I-{label.replace('B-', '')}"
#                         labels.append(lab)
#                         total_labels.append(lab)
#                 elif 'I' == label[0]:
#                     labels.append(label)
#                     total_labels.append(label)
#                 elif 'E' == label[0]:
#                     lab = f"I-{label.replace('E-', '')}"
#                     labels.append(lab)
#                     total_labels.append(lab)
#                 else:
#                     print(label)
#                     raise TypeError('not supported label type!')
#         else:
#             assert len(labels) == len(sequences)
#             D.append((sequences, labels))
#             sequences = []
#             labels = []


# ## Conll2003
# with open('raw_data/Conll2003/dev.txt', encoding='utf-8') as f:
#     for line in f.readlines():
#         sequences = []
#         labels = []
#         line=json.loads(line.strip())
#         texts = line['text'].split()
#         label_raw = ['O'] * len(texts)
#         if line['label'] != {}:
#             for lab_type, ents in line['label'].items():
#                 for ent, span in ents.items():
#                     span = span[0]
#                     for idx_span, j in enumerate(range(span[0], span[1]+1)):
#                         if idx_span == 0:
#                             label_raw[j] = f'B-{lab_type}'
#                         else:
#                             label_raw[j] = f'I-{lab_type}'
#
#         for words, label in zip(texts, label_raw):
#             tokenized_words = tokenizer.tokenize(words)  # 对每个word tokenize
#             for i, word in enumerate(tokenized_words):  # 如果一个word被拆分成了多个token
#                 sequences.append(word)
#                 if label == 'O':
#                     labels.append(label)
#                     total_labels.append(label)
#                 elif 'S' == label[0]:
#                     if i == 0:
#                         lab = f"B-{label.replace('S-', '')}"
#                         labels.append(lab)
#                         total_labels.append(lab)
#                     else:
#                         lab = f"I-{label.replace('S-', '')}"
#                         labels.append(lab)
#                         total_labels.append(lab)
#                 elif 'B' == label[0]:  # subword第一个为B，其余的为I
#                     if i == 0:
#                         lab = f"B-{label.replace('B-', '')}"
#                         labels.append(lab)
#                         total_labels.append(lab)
#                     else:
#                         lab = f"I-{label.replace('B-', '')}"
#                         labels.append(lab)
#                         total_labels.append(lab)
#                 elif 'I' == label[0]:
#                     labels.append(label)
#                     total_labels.append(label)
#                 elif 'E' == label[0]:
#                     lab = f"I-{label.replace('E-', '')}"
#                     labels.append(lab)
#                     total_labels.append(lab)
#                 else:
#                     print(label)
#                     raise TypeError('not supported label type!')
#
#         assert len(labels) == len(sequences)
#         D.append([sequences, labels])


print(len(D))
id2lab = {i:k for i,k in enumerate(list(set(total_labels)))}
print(id2lab)
# with open('processed_data/twitter2017/train.json', 'w', encoding='utf-8') as f:
#     for line in D:
#         f.write(json.dumps(line)+'\n')

# random.shuffle(D)
# with open('processed_data/SciTech/train.json', 'w', encoding='utf-8') as f:
#     for line in D[:600]:
#         f.write(json.dumps(line)+'\n')
#
#
# with open('processed_data/SciTech/dev.json', 'w', encoding='utf-8') as f:
#     for line in D[600:1200]:
#         f.write(json.dumps(line)+'\n')
#
# with open('processed_data/SciTech/test.json', 'w', encoding='utf-8') as f:
#     for line in D[1200:]:
#         f.write(json.dumps(line)+'\n')



