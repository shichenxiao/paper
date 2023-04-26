import torch
import torch.nn as nn
import os
from . import utils as solver_utils 
from utils.utils import to_cuda, mean_accuracy, accuracy,accuracy_test
from seqeval.metrics import classification_report
from torch import optim
import numpy as np
from math import ceil as ceil
from config.config import cfg
from data.single_dataset import idx2tag,tag2idx
from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics
from solver.metrics import Metrics
from solver.evaluate import Precision,Recall,F1_score
# from solver.evaluate import precision,recall,f1_score
# id2label = {0: 'O', 1: 'I-PER', 2: 'B-PER', 3: 'I-ORG', 4: 'B-LOC', 5: 'I-MISC', 6: 'B-MISC', 7: 'I-LOC', 8: 'B-ORG'}
id2label = {0:'B-LOC', 1:'I-LOC', 2:'B-PER',
        3:'I-PER', 4:'B-ORG', 5:'I-ORG', 6:'B-MISC',7:'I-MISC',8:'O'}
label2id = {v:k for k,v in id2label.items()}
class BaseSolver:
    def __init__(self, net, dataloader, bn_domain_map={}, resume=None, **kwargs):
        self.opt = cfg
        self.source_name = self.opt.DATASET.SOURCE_NAME
        self.target_name = self.opt.DATASET.TARGET_NAME
        self.target = self.opt.DATASET.TARGET_NAME+'_dev'
        self.source = self.opt.DATASET.SOURCE_NAME+'_dev'
        self.net = net
        self.init_data(dataloader)

        self.CELoss = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            self.CELoss.cuda() 

        self.loop = 0
        self.iters = 0
        self.iters_per_loop = None
        self.history = {}

        self.base_lr = self.opt.TRAIN.BASE_LR
        self.momentum = self.opt.TRAIN.MOMENTUM

        self.bn_domain_map = bn_domain_map

        self.optim_state_dict = None
        self.resume = False
        if resume is not None:
            self.resume = True
            self.loop = resume['loop']
            self.iters = resume['iters']
            self.history = resume['history']
            self.optim_state_dict = resume['optimizer_state_dict']
            self.bn_domain_map = resume['bn_domain_map']
            print('Resume Training from loop %d, iters %d.' % \
			(self.loop, self.iters))

        # self.build_optimizer()

    def init_data(self, dataloader):
        self.train_data = {key: dict() for key in dataloader if key != 'test'}
        for key in self.train_data.keys():
            if key not in dataloader:
                continue
            cur_dataloader = dataloader[key]
            self.train_data[key]['loader'] = cur_dataloader 
            self.train_data[key]['iterator'] = None

        if 'test' in dataloader:
            self.test_data = dict()
            self.test_data['loader'] = dataloader['test']
        
    # def build_optimizer(self):
    #     opt = self.opt
    #     param_groups = solver_utils.set_param_groups(self.net,
	# 	dict({'FC': opt.TRAIN.LR_MULT}))
    #
    #     assert opt.TRAIN.OPTIMIZER in ["Adam", "SGD"], \
    #         "Currently do not support your specified optimizer."
    #
    #     if opt.TRAIN.OPTIMIZER == "Adam":
    #         self.optimizer = optim.Adam(param_groups,
	# 		lr=self.base_lr, betas=[opt.ADAM.BETA1, opt.ADAM.BETA2],
	# 		weight_decay=opt.TRAIN.WEIGHT_DECAY)
    #
    #     elif opt.TRAIN.OPTIMIZER == "SGD":
    #         self.optimizer = optim.SGD(param_groups,
	# 		lr=self.base_lr, momentum=self.momentum,
	# 		weight_decay=opt.TRAIN.WEIGHT_DECAY)
    #
    #     if self.optim_state_dict is not None:
    #         self.optimizer.load_state_dict(self.optim_state_dict)

    def update_lr(self):
        iters = self.iters
        if self.opt.TRAIN.LR_SCHEDULE == 'exp':
            solver_utils.adjust_learning_rate_exp(self.base_lr, 
			self.optimizer, iters, 
                        decay_rate=self.opt.EXP.LR_DECAY_RATE,
			decay_step=self.opt.EXP.LR_DECAY_STEP)

        elif self.opt.TRAIN.LR_SCHEDULE == 'inv':
            solver_utils.adjust_learning_rate_inv(self.base_lr, self.optimizer, 
		    iters, self.opt.INV.ALPHA, self.opt.INV.BETA)

        else:
            raise NotImplementedError("Currently don't support the specified \
                    learning rate schedule: %s." % self.opt.TRAIN.LR_SCHEDULE)

    def logging(self, loss, accu):
        print('[loop: %d, iters: %d]: ' % (self.loop, self.iters))
        loss_names = ""
        loss_values = ""
        for key in loss:
            loss_names += key + ","
            loss_values += '%.4f,' % (loss[key])
        loss_names = loss_names[:-1] + ': '
        loss_values = loss_values[:-1] + ';'
        loss_str = loss_names + loss_values + (' source %s: %.4f.' % 
                    (self.opt.EVAL_METRIC, accu))
        print(loss_str)
    def _evaluate(self, do_metric, data_loader):
        self.model.eval()
        eval_results = {}
        total_pred = []
        total_text = []
        total_res = []

        with torch.no_grad():
            # start_time = time.time()
            # data_loader = tqdm(data_loader, leave=False)
            for batch in data_loader:
                # _move_dict_value_to_device(batch, device=self.device)
                outputs = self._data_forward_eval(self._evaluate_func, batch)

                if do_metric:
                    res = self.compute_metric(outputs, batch)
                    total_res.append(res)
                    if not eval_results:
                        for k, v in res.items():
                            eval_results[k] = [v]
                    else:
                        for k, v in res.items():
                            eval_results[k].append(v)
                else:
                    total_pred.extend(outputs.logits)
                    if 'text' in batch:
                        total_text.extend(batch['text'])

            if do_metric:
                for k, v in eval_results.items():
                    eval_results[k] = round(sum(eval_results[k]) / len(eval_results[k]), 5)

            # end_time = time.time()
            # test_str = f'Evaluate data in {round(end_time - start_time, 2)} seconds!'
            # logger.info(test_str)

        if do_metric:
            return eval_results
        else:
            if total_text == []:
                return total_pred
            else:
                return zip(total_pred, total_text)

    def model_eval(self, model, iterator):
        model.eval()

        Words, Tags, Y, Y_hat = [], [], [], []
        with torch.no_grad():
            for i, batch in enumerate(iterator):
                words = batch['sentences']
                x = batch['word']
                # is_heads = batch['is_head']
                tags = batch['tags']
                y = batch['Label']
                seqlens = batch['seqlens']
                x = x.cuda()
                # y = y.cuda()
                mask = (x != 0).cuda()
                y_hat = model.predict(x, mask)  # y_hat: (N, T)
                t = []
                Tags_temp=[]
                tags_temp = []
                for tag in tags:
                    t.append(tag.split(' '))
                # tags_temp.append(t)
                Words.extend(words)
                # Is_heads.extend(is_heads)
                Tags_temp.extend(t)
                Y.extend(y.numpy().tolist())
                labels_lst =[]
                labels = []
                for lst in Tags_temp:
                    for tag in lst:
                        labels.append(tag2idx.get(tag))
                    labels_lst.append(labels)
                    labels=[]
                temps = []
                for lst in y_hat:
                    temp = []
                    for l in lst:
                        temp.append(idx2tag.get(l))
                    temps.append(temp)
                Y_hat.extend(y_hat)
                Tags.extend(labels_lst)
        res = self.compute_metric(Y_hat, Tags)
        # print('111')
        return res
        # total_res.append(res)
        # if not eval_results:
        #     for k, v in res.items():
        #         eval_results[k] = [v]
        # else:
        #     for k, v in res.items():
        #         eval_results[k].append(v)
        # Metrics(Tags, Y_hat)
        #     for k, v in eval_results.items():
        #         eval_results[k] = round(sum(eval_results[k]) / len(eval_results[k]), 5)
        # return eval_results
        # precision = Precision(Y_hat,Tags)
        # recall = Recall(Y_hat,Tags)
        # f1_score = F1_score(precision, recall)
            ## 暂存结果
        # with open("temp", 'w', encoding='utf-8') as fout:
        #     for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
        #         y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
        #         preds = [idx2tag[hat] for hat in y_hat]
        #         assert len(preds) == len(words.split()) == len(tags.split())
        #         for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
        #             fout.write(f"{w} {t} {p}\n")
        #         fout.write("\n")
        #
        # ## calc metric
        # y_true = np.array(
        #     [tag2idx[line.split()[1]] for line in open("temp", 'r', encoding='utf-8').read().splitlines() if
        #      len(line) > 0])
        # y_pred = np.array(
        #     [tag2idx[line.split()[2]] for line in open("temp", 'r', encoding='utf-8').read().splitlines() if
        #      len(line) > 0])
        # tp, fp, fn = 0, 0, 0
        # for i in range(len(y_true)):
        #     if y_true[i] == y_pred[i]:
        #         if y_true[i] != 3:
        #             tp += 1
        #     else:
        #         if y_true[i] != 3 and y_pred[i] != 3:
        #             fp += 1
        #             fn += 1
        #         elif y_true[i] != 3 and y_pred[i] == 3:
        #             fn += 1
        #         elif y_true[i] == 3 and y_pred[i] != 3:
        #             fp += 1
        # precision = tp / (tp + fp)
        # recall = tp / (tp + fn)
        # f1 = 2 * precision * recall / (precision + recall)
        # print("precision=%.5f" % precision)
        # print("recall=%.5f" % recall)
        # print("f1=%.5f" % f1)
        # return precision, recall, f1
        # num_proposed = len(y_pred[y_pred > 1])
        # num_correct = (np.logical_and(y_true == y_pred, y_true > 1)).astype(np.int).sum()
        # num_gold = len(y_true[y_true > 1])
        #
        # print(f"num_proposed:{num_proposed}")
        # print(f"num_correct:{num_correct}")
        # print(f"num_gold:{num_gold}")
        # try:
        #     precision = num_correct / num_proposed
        # except ZeroDivisionError:
        #     precision = 1.0
        #
        # try:
        #     recall = num_correct / num_gold
        # except ZeroDivisionError:
        #     recall = 1.0
        #
        # try:
        #     f1 = 2 * precision * recall / (precision + recall)
        # except ZeroDivisionError:
        #     if precision * recall == 0:
        #         f1 = 1.0
        #     else:
        #         f1 = 0
        # #
        # # final = f + ".P%.2f_R%.2f_F%.2f" % (precision, recall, f1)
        # # with open(final, 'w', encoding='utf-8') as fout:
        # #     result = open("temp", "r", encoding='utf-8').read()
        # #     fout.write(f"{result}\n")
        # #
        # #     fout.write(f"precision={precision}\n")
        # #     fout.write(f"recall={recall}\n")
        # #     fout.write(f"f1={f1}\n")
        # #
        # # os.remove("temp")
        # # precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
        # # precision = metrics.precision_score(y_true, y_pred, average='macro')
        # # recall = metrics.recall_score(y_true, y_pred, average='macro')
        # # f1_score = metrics.f1_score(y_true, y_pred, average='macro')
        # print("precision=%.5f" % precision)
        # print("recall=%.5f" % recall)
        # print("f1=%.5f" % f1_score)
        #
        # return precision, recall,f1_score

    def compute_metric(self,pred, batch):
        # logits = torch.argmax(pred.logits, dim=2).tolist()
        logits = pred
        # labels = batch['Label']
        labels = batch
        t=[]
        tags_temp = []
        # for tag in tags:
        #     t.append(tag.split(' '))
        # labels = []
        # temp = []
        # for lst in t:
        #     for tag in lst:
        #          temp.append(label2id.get(tag))
        #     labels.append(temp)
        #     temp=[]
        out_label_list = [[] for _ in range(len(labels))]
        preds_list = [[] for _ in range(len(labels))]
        total_label = []
        total_pred = []
        for i in range(len(labels)):  # pad token label id 对应的预测结果是不需要的
            for j in range(len(labels[i])):
                if  len(labels[i]) == len(logits[i]) and labels[i][j]!=8:
                # if labels[i][j] != label2id['O'] and labels[i][j] != 9 and labels[i][j] != 10 and labels[i][j] != 11:
                    out_label_list[i].append(idx2tag[labels[i][j]])
                    # print(id2label[logits[i][j]])
                    # print()
                    preds_list[i].append(idx2tag[logits[i][j]])
                    total_label.append(idx2tag[labels[i][j]])
                    total_pred.append(idx2tag[logits[i][j]])

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
        # results.update({'loss': pred.loss.item()})
        return results
    def save_ckpt(self):
        save_path = self.opt.SAVE_DIR
        print('1111111111'+save_path)
        ckpt_resume = os.path.join(save_path, 'ckpt_%d_%d.resume' % (self.loop, self.iters))
        print(ckpt_resume)
        ckpt_weights = os.path.join(save_path, 'ckpt_%d_%d.weights' % (self.loop, self.iters))
        
        torch.save({'loop': self.loop,
                    'iters': self.iters,
                    'model_state_dict': self.net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'history': self.history,
                    'bn_domain_map': self.bn_domain_map
                    }, ckpt_resume)

        torch.save({'weights': self.net.state_dict(),
                    'bn_domain_map': self.bn_domain_map
                    }, ckpt_weights)

    def complete_training(self):
        if self.loop > self.opt.TRAIN.MAX_LOOP:
            return True

    def register_history(self, key, value, history_len):
        if key not in self.history:
            self.history[key] = [value]
        else:
            self.history[key] += [value]
        
        if len(self.history[key]) > history_len:
            self.history[key] = \
                 self.history[key][len(self.history[key]) - history_len:]
       
    def solve(self):
        print('Training Done!')

    def get_samples(self, data_name):
        assert(data_name in self.train_data)
        assert('loader' in self.train_data[data_name] and \
               'iterator' in self.train_data[data_name])

        data_loader = self.train_data[data_name]['loader']
        data_iterator = self.train_data[data_name]['iterator']
        assert data_loader is not None and data_iterator is not None, \
            'Check your dataloader of %s.' % data_name 

        try:
            sample = next(data_iterator)
        except StopIteration:
            data_iterator = iter(data_loader)
            sample = next(data_iterator)
            self.train_data[data_name]['iterator'] = data_iterator
        return sample

    def get_samples_categorical(self, data_name, category):
        assert(data_name in self.train_data)
        assert('loader' in self.train_data[data_name] and \
               'iterator' in self.train_data[data_name])

        data_loader = self.train_data[data_name]['loader'][category]
        data_iterator = self.train_data[data_name]['iterator'][category]
        assert data_loader is not None and data_iterator is not None, \
            'Check your dataloader of %s.' % data_name

        try:
            sample = next(data_iterator)
        except StopIteration:
            data_iterator = iter(data_loader)
            sample = next(data_iterator)
            self.train_data[data_name]['iterator'][category] = data_iterator

        return sample

    def test(self):
        self.net.eval()
        preds = []
        gts = []
        is_heads =[]
        words = []
        tags = []
        for sample in iter(self.test_data['loader']):
            is_heads.extend(sample['is_head'])
            tags.extend(sample['tags'])
            words.extend(sample['sentences'])
            data, gt = to_cuda(sample['word']), to_cuda(sample['Label'])
            logits = self.net(data)['logits']
            preds.extend(logits)
            gts.extend(gts)

        # preds = torch.cat(preds, dim=0)
        # gts = torch.cat(gts, dim=0)

        res = accuracy_test(preds, tags,is_heads,words)
        return res

    def clear_history(self, key):
        if key in self.history:
            self.history[key].clear()

    def solve(self):
        pass

    def update_network(self, **kwargs):
        pass

