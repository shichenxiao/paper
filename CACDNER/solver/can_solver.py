import torch
import torch.nn as nn
import os
from . import utils as solver_utils
from utils.utils import to_cuda, to_onehot
from torch import optim
from . import clustering
from discrepancy.cdd import CDD
from math import ceil as ceil
from .base_solver import BaseSolver
from copy import deepcopy
from ordered_set import OrderedSet
from torch.autograd import Variable
from discrepancy.djp_mmd import rbf_djpmmd,rbf_jpmmd,rbf_jmmd,rbf_mmd
import time
from discrepancy.JPDA_compare_python import DA_statistics
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if torch.cuda.is_available():
    device = torch.device("cuda", 0)
    print('device',device)
    use_cuda = True
GAMMA = 1000  # 1000 more weight to transferability
SIGMA = 1  # default 1
class CANSolver(BaseSolver):
    def __init__(self, net, dataloader, bn_domain_map={}, resume=None, **kwargs):
        super(CANSolver, self).__init__(net, dataloader, \
                                        bn_domain_map=bn_domain_map, resume=resume, **kwargs)

        if len(self.bn_domain_map) == 0:
            self.bn_domain_map = {self.source_name: 0, self.target_name: 1}

        self.clustering_source_name = 'clustering_' + self.source_name
        self.clustering_target_name = 'clustering_' + self.target_name

        assert ('categorical' in self.train_data)

        # num_layers = len(self.net.FC) + 1
        num_layers=2
        self.cdd = CDD(kernel_num=self.opt.CDD.KERNEL_NUM, kernel_mul=self.opt.CDD.KERNEL_MUL,
                       num_layers=num_layers, num_classes=self.opt.DATASET.NUM_CLASSES,
                       intra_only=self.opt.CDD.INTRA_ONLY)

        self.discrepancy_key = 'intra' if self.opt.CDD.INTRA_ONLY else 'cdd'
        self.clustering = clustering.Clustering(self.opt.CLUSTERING.EPS,
                                                self.opt.CLUSTERING.FEAT_KEY,
                                                self.opt.CLUSTERING.BUDGET)

        self.clustered_target_samples = {}
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.opt.TRAIN.BASE_LR)
        self.DA_statistics = DA_statistics(kernel_type='primal', mmd_type='djp-mmd', dim=30, lamb=1, gamma=1, mu=0.1, T=5)

    def complete_training(self):
        if self.loop >= self.opt.TRAIN.MAX_LOOP:
            return True

        if 'target_centers' not in self.history or \
                'ts_center_dist' not in self.history or \
                'target_labels' not in self.history:
            return False

        if len(self.history['target_centers']) < 2 or \
                len(self.history['ts_center_dist']) < 1 or \
                len(self.history['target_labels']) < 2:
            return False

        # target centers along training
        target_centers = self.history['target_centers']
        eval1 = torch.mean(self.clustering.Dist.get_dist(target_centers[-1],
                                                         target_centers[-2])).item()

        # target-source center distances along training
        eval2 = self.history['ts_center_dist'][-1].item()

        # target labels along training
        path2label_hist = self.history['target_labels']
        paths = self.clustered_target_samples['data']
        num = 0
        for path in paths:
            pre_label = path2label_hist[-2]
            cur_label = path2label_hist[-1]
            if pre_label != cur_label:
                num += 1
        eval3 = 1.0 * num / len(paths)

        return (eval1 < self.opt.TRAIN.STOP_THRESHOLDS[0] and \
                eval2 < self.opt.TRAIN.STOP_THRESHOLDS[1] and \
                eval3 < self.opt.TRAIN.STOP_THRESHOLDS[2])

    def solve(self):
        stop = False
        if self.resume:
            self.iters += 1
            self.loop += 1

        while True:
            # updating the target label hypothesis through clustering
            target_hypt = {}
            filtered_classes = []
            with torch.no_grad():
                # self.update_ss_alignment_loss_weight()
                print('Clustering based on %s...' % self.source_name)
                self.update_labels()
                self.clustered_target_samples = self.clustering.samples
                target_centers = self.clustering.centers
                center_change = self.clustering.center_change
                path2label = self.clustering.path2label

                # updating the history
                self.register_history('target_centers', target_centers,
                                      self.opt.CLUSTERING.HISTORY_LEN)
                self.register_history('ts_center_dist', center_change,
                                      self.opt.CLUSTERING.HISTORY_LEN)
                self.register_history('target_labels', path2label,
                                      self.opt.CLUSTERING.HISTORY_LEN)

                if self.clustered_target_samples is not None and \
                        self.clustered_target_samples['gt'] is not None:
                    preds = to_onehot(self.clustered_target_samples['label'],
                                      self.opt.DATASET.NUM_CLASSES)
                    gts = self.clustered_target_samples['gt']
                    res = self.model_eval(preds, gts)
                    print('Clustering %s: %.4f' % (self.opt.EVAL_METRIC, res))

                # check if meet the stop condition
                stop = self.complete_training()
                if stop: break

                # filtering the clustering results
                target_sample= self.filtering()

                # update dataloaders
                # self.construct_categorical_dataloader(target_hypt, filtered_classes)
                self.construct_categorical_dataloader(target_sample)
                # update train data setting
                # self.compute_iters_per_loop(filtered_classes)

            # k-step update of network parameters through forward-backward process
            self.update_network(filtered_classes)

            self.loop += 1

        print('Training Done!')

    def update_labels(self):
        net = self.net
        net.eval()
        opt = self.opt

        source_dataloader = self.train_data[self.clustering_source_name]['loader']
        # net.set_bn_domain(self.bn_domain_map[self.source_name])

        source_centers = solver_utils.get_centers(net,
                                                  source_dataloader, self.opt.DATASET.NUM_CLASSES,self.opt.CLUSTERING.FEAT_KEY)
        init_target_centers = source_centers

        target_dataloader = self.train_data[self.clustering_target_name]['loader']
        # net.set_bn_domain(self.bn_domain_map[self.target_name])

        self.clustering.set_init_centers(init_target_centers)
        self.clustering.feature_clustering(net, target_dataloader)

    def filtering(self):
        threshold = self.opt.CLUSTERING.FILTERING_THRESHOLD
        min_sn_cls = self.opt.TRAIN.MIN_SN_PER_CLASS
        target_samples = self.clustered_target_samples

        # filtering the samples
        # chosen_samples = solver_utils.filter_samples(
        #     target_samples, threshold=threshold)

        # filtering the classes
        # filtered_classes = solver_utils.filter_class(
        #     chosen_samples['label'], min_sn_cls, self.opt.DATASET.NUM_CLASSES)

        # print('The number of filtered classes: %d.' % len(filtered_classes))
        return target_samples

    def construct_target_dataloader(self,samples):
        dataloader =self.train_data[self.clustering_source_name]['loader']

        dataloader.construct()
    def construct_categorical_dataloader(self, samples):
        # update self.dataloader
        # target_classwise = solver_utils.split_samples_classwise(
        #     samples, self.opt.DATASET.NUM_CLASSES)

        dataloader = self.train_data['categorical']['loader']
        dataloader.target_labels = samples['label']
        dataloader.target_data = samples['data']
        # classnames = dataloader.classnames
        # dataloader.class_set = [classnames[c] for c in filtered_classes]
        # dataloader.target_paths = {classnames[c]: target_classwise[c]['datas'] \
        #                            for c in filtered_classes}
        # dataloader.num_selected_classes = min(self.opt.TRAIN.NUM_SELECTED_CLASSES, len(filtered_classes))
        dataloader.construct()

    def CAS(self):
        samples = self.get_samples('categorical')

        source_samples = samples['Img_source']
        source_sample_paths = samples['Path_source']
        source_nums = [len(paths) for paths in source_sample_paths]

        target_samples = samples['Img_target']
        target_sample_paths = samples['Path_target']
        target_nums = [len(paths) for paths in target_sample_paths]

        source_sample_labels = samples['Label_source']
        self.selected_classes = [labels[0].item() for labels in source_sample_labels]
        assert (self.selected_classes ==
                [labels[0].item() for labels in samples['Label_target']])
        return source_samples, source_nums, target_samples, target_nums

    def prepare_feats(self, feats):
        return [feats[key] for key in feats if key in self.opt.CDD.ALIGNMENT_FEAT_KEYS]

    def compute_iters_per_loop(self, filtered_classes):
        self.iters_per_loop = int(
            len(self.train_data['categorical']['loader'])) * self.opt.TRAIN.UPDATE_EPOCH_PERCENTAGE
        print('Iterations in one loop: %d' % (self.iters_per_loop))

    def mmd_loss(self,x_src, y_src, x_tar, y_pseudo, mmd_type):
        if mmd_type == 'mmd':
            return rbf_mmd(x_src, x_tar, SIGMA)
        elif mmd_type == 'jmmd':
            return rbf_jmmd(x_src, y_src, x_tar, y_pseudo, SIGMA)
        elif mmd_type == 'jpmmd':
            return rbf_jpmmd(x_src, y_src, x_tar, y_pseudo, SIGMA)
        elif mmd_type == 'djpmmd':
            return rbf_djpmmd(x_src, y_src, x_tar, y_pseudo, SIGMA)
    def update_network(self, filtered_classes):
        # initial configuration

        # while not stop:
        # update learning rate
        # self.update_lr()

        # set the status of network
        self.net.train()
        # self.net.zero_grad()

        loss = 0

        # coventional sampling for training on labeled source data
        # source_sample = self.get_samples(self.source_name)
        # source_data, source_gt = source_sample['word'], \
        #                          source_sample['Label']
        # sorce_isHead,source_sentence = source_sample['is_head'],\
        #                                 source_sample['sentences']
        self.train_data[self.source_name]['iterator'] = \
            iter(self.train_data[self.source_name]['loader'])
        self.train_data[self.source]['iterator'] = \
            iter(self.train_data[self.source]['loader'])
        self.train_data[self.target]['iterator'] = \
            iter(self.train_data[self.target]['loader'])
        self.train_data['categorical']['iterator'] = \
            iter(self.train_data['categorical']['loader'])
        self.train_data[self.target_name]['iterator'] = \
            iter(self.train_data[self.target_name]['loader'])
        iterator = self.train_data[self.source_name]['loader']

        res = self.model_eval(self.net, self.train_data[self.target_name]['iterator'])
        for i,source_sample in enumerate(iterator):
            source_data, source_gt = source_sample['word'], \
                                     source_sample['Label']

            source_data = to_cuda(source_data)
            source_gt = to_cuda(source_gt)  #[30,50]
            mask = source_data != 0
            self.net.set_bn_domain(self.bn_domain_map[self.source_name])
            self.net.zero_grad()
            source_preds,_ = self.net.get_lstm_features(source_data) #[30,50,64]
            loss_model = self.net(source_data, source_gt,mask)
            # loss_model.backward()
            target_sample = self.get_samples('categorical')
            target_data, target_gt = target_sample['words'], \
                                     target_sample['label']

            target_data = to_cuda(target_data)

            target_preds,_ = self.net.get_lstm_features(target_data)
            # target_label = target_label
            loss_mmd = self.mmd_loss(source_preds, source_gt, target_preds, target_gt, 'djpmmd')

            # loss_mmd.backward()
            # target_label = self.DA_statistics.fit_predict(source_preds,source_gt,target_preds,target_gt)
            # mask1 = target_data != 0
            # loss_target = self.net(target_preds, target_label, mask1)
            # loss = loss_source+loss_target
            loss = loss_mmd*1000+loss_model
            loss.backward()
            if i%10==0:
                logger.info('model_loss:{},mmd_loss:{},total_loss:{},iters:{}'.format(loss_model,loss_mmd,loss,i))
                # logger.info('model_loss:{}'.format(loss_model))
            # update the network
            self.optimizer.step()
        fname = os.path.join(self.opt.EXP_NAME, str(self.iters))
        torch.save(self.net.state_dict(),f"{fname}.pt")
        print(f"=========eval at lters{self.iters}==========")
        print(f"source_data:")
        res=self.model_eval(self.net,self.train_data[self.source]['iterator'])
        print(res)
        print(f"target_data:")
        res=self.model_eval(self.net,self.train_data[self.target]['iterator'])
        print(res)
        self.iters += 1


