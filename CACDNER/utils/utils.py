import torch
import numpy as np
from data.single_dataset import idx2tag,tag2idx
def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)

def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()

def to_onehot(label, num_classes):
    identity = to_cuda(torch.eye(num_classes))
    onehot = torch.index_select(identity, 0, label)
    return onehot

def mean_accuracy(preds, target):
    num_classes = preds.size(1)
    preds = torch.max(preds, dim=1).indices
    accu_class = []
    for c in range(num_classes):
        mask = (target == c)
        c_count = torch.sum(mask).item()
        if c_count == 0: continue
        preds_c = torch.masked_select(preds, mask)
        accu_class += [1.0 * torch.sum(preds_c == c).item() / c_count]
    return 100.0 * np.mean(accu_class)

def accuracy(pred, target,Is_heads,Words):
    pred =pred.argmax(-1)
    Y_hat = []
    Y_hat.extend(pred.cpu().numpy().tolist())

    # Tags = target
    # preds = torch.max(preds, dim=1).indices
    # return 100.0 * torch.sum(preds == target).item() / preds.size(0)
    with open("temp", 'w', encoding='utf-8') as fout:
        for words, is_heads, tags, y_hat in zip(Words, Is_heads, target, Y_hat):
            y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
            preds = [idx2tag[hat] for hat in y_hat]
            # if(len(preds) != len(words.split()))
            assert len(preds) == len(words.split()) == len(tags.split())
            for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
                fout.write(f"{w} {t} {p}\n")
            fout.write("\n")

    ## calc metric
    y_true = np.array(
        [tag2idx[line.split()[1]] for line in open("temp", 'r', encoding='utf-8').read().splitlines() if len(line) > 0])
    y_pred = np.array(
        [tag2idx[line.split()[2]] for line in open("temp", 'r', encoding='utf-8').read().splitlines() if len(line) > 0])

    num_proposed = len(y_pred[y_pred > 1])
    num_correct = (np.logical_and(y_true == y_pred, y_true > 1)).astype(np.int).sum()
    num_gold = len(y_true[y_true > 1])

    print(f"num_proposed:{num_proposed}")
    print(f"num_correct:{num_correct}")
    print(f"num_gold:{num_gold}")
    try:
        precision = num_correct / num_proposed
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = num_correct / num_gold
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        if precision * recall == 0:
            f1 = 1.0
        else:
            f1 = 0

    # final = f + ".P%.2f_R%.2f_F%.2f" % (precision, recall, f1)
    # with open(final, 'w', encoding='utf-8') as fout:
    #     result = open("temp", "r", encoding='utf-8').read()
    #     fout.write(f"{result}\n")
    #
    #     fout.write(f"precision={precision}\n")
    #     fout.write(f"recall={recall}\n")
    #     fout.write(f"f1={f1}\n")
    #
    # os.remove("temp")

    print("precision=%.2f" % precision)
    print("recall=%.2f" % recall)
    print("f1=%.2f" % f1)
    return precision, recall, f1

def accuracy_test(pred, Tags,Is_heads,Words):
    Y_hat = []
    l = len(pred)
    for i in range(l):
        pred[i]=pred[i].argmax(-1).cpu().numpy().tolist()
        # y = y.to(device)
        # Y.extend(y.numpy().tolist())
        # Y_hat.extend(x.cpu().numpy().tolist())
    # Tags = target
    # preds = torch.max(preds, dim=1).indices
    # return 100.0 * torch.sum(preds == target).item() / preds.size(0)
    with open("temp", 'w', encoding='utf-8') as fout:
        for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, pred):
            y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
            preds = [idx2tag[hat] for hat in y_hat]
            # if(len(preds) != len(words.split()))
            assert len(preds) == len(words.split()) == len(tags.split())
            for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
                fout.write(f"{w} {t} {p}\n")
            fout.write("\n")

    ## calc metric
    y_true = np.array(
        [tag2idx[line.split()[1]] for line in open("temp", 'r', encoding='utf-8').read().splitlines() if len(line) > 0])
    y_pred = np.array(
        [tag2idx[line.split()[2]] for line in open("temp", 'r', encoding='utf-8').read().splitlines() if len(line) > 0])

    num_proposed = len(y_pred[y_pred > 1])
    num_correct = (np.logical_and(y_true == y_pred, y_true > 1)).astype(np.int).sum()
    num_gold = len(y_true[y_true > 1])

    print(f"num_proposed:{num_proposed}")
    print(f"num_correct:{num_correct}")
    print(f"num_gold:{num_gold}")
    try:
        precision = num_correct / num_proposed
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = num_correct / num_gold
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        if precision * recall == 0:
            f1 = 1.0
        else:
            f1 = 0

    # final = f + ".P%.2f_R%.2f_F%.2f" % (precision, recall, f1)
    # with open(final, 'w', encoding='utf-8') as fout:
    #     result = open("temp", "r", encoding='utf-8').read()
    #     fout.write(f"{result}\n")
    #
    #     fout.write(f"precision={precision}\n")
    #     fout.write(f"recall={recall}\n")
    #     fout.write(f"f1={f1}\n")
    #
    # os.remove("temp")

    print("precision=%.2f" % precision)
    print("recall=%.2f" % recall)
    print("f1=%.2f" % f1)
    return precision, recall, f1