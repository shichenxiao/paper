import torch
from torch.nn import functional as F
from utils.utils import to_cuda, to_onehot
from scipy.optimize import linear_sum_assignment
from math import ceil


class DIST(object):
    def __init__(self, dist_type='cos'):
        self.dist_type = dist_type

    def get_dist(self, pointA, pointB, cross=False):
        return getattr(self, self.dist_type)(
            pointA, pointB, cross)

    def cos(self, pointA, pointB, cross):
        pointA = F.normalize(pointA, dim=1)
        pointB = F.normalize(pointB, dim=1)
        if not cross:
            return 0.5 * (1.0 - torch.sum(pointA * pointB, dim=1))
        else:
            NA = pointA.size(0)
            NB = pointB.size(0)
            assert (pointA.size(1) == pointB.size(1))
            return 0.5 * (1.0 - torch.matmul(pointA, pointB.transpose(0, 1)))


class Clustering(object):
    def __init__(self, eps, feat_key, max_len=100, dist_type='cos'):
        self.eps = eps
        self.Dist = DIST(dist_type)
        self.samples = {}
        self.path2label = []
        self.character2label = {}
        self.center_change = None
        self.stop = False
        self.feat_key = feat_key
        self.max_len = max_len

    def set_init_centers(self, init_centers):
        self.centers = init_centers
        self.init_centers = init_centers
        self.num_classes = self.centers.size(0)

    def clustering_stop(self, centers):
        if centers is None:
            self.stop = False
        else:
            dist = self.Dist.get_dist(centers, self.centers)
            dist = torch.mean(dist, dim=0)
            print('dist %.4f' % dist.item())
            self.stop = dist.item() < self.eps

    def assign_labels1(self, feats):
        dists = self.Dist.get_dist(feats, self.centers, cross=True)
        _, labels = torch.min(dists, dim=2)
        return dists, labels

    def assign_labels(self, feats):
        dists = self.Dist.get_dist(feats, self.centers, cross=True)
        _, labels = torch.min(dists, dim=1)
        return dists, labels

    def align_centers(self):
        cost = self.Dist.get_dist(self.centers, self.init_centers, cross=True)
        cost = cost.data.cpu().numpy()
        _, col_ind = linear_sum_assignment(cost)
        return col_ind

    def collect_samples(self, net, loader):
        data_feat, data_gt = [], []
        data_paths, datas ,data_feat_use= [], [],[]
        is_head = []
        max_len=0
        for sample in iter(loader):
            data = sample['word'].cuda()

            data_paths += sample['word'].tolist()
            # is_head += sample['is_head']
            if 'Label' in sample.keys():
                data_gt += [to_cuda(sample['Label'])]

            output,_ = net.get_lstm_features(data)
            feature = output.data
            output1 = torch.reshape(feature, (-1, 768))
            # feature = output[self.feat_key].data  #(100, 2048)
            data_feat += [feature]
            data_feat_use += [output1]
            if max_len<data.size(1):
                max_len = data.size(1)
        lens = data_paths.__len__()
        for i in range(lens):
            datas += data_paths[i]
        self.samples['data'] = data_paths
        # self.samples['is_head'] = is_head
        self.samples['gt'] = torch.cat(data_gt, dim=0) \
            if len(data_gt) > 0 else None
        self.samples['feature'] = torch.cat(data_feat, dim=0)
        # self.samples['feature'] = data_feat
        self.samples['feature_use'] = torch.cat(data_feat_use, dim=0)

    def feature_clustering(self, net, loader):
        centers = None
        self.stop = False
        loop=0
        self.collect_samples(net,loader)
        feature = self.samples['feature']# (498,2048)  source data 498条
        # feature_use =self.samples['feature_use']
        # is_heads =self.samples['is_head']
        refs = to_cuda(torch.LongTensor(range(self.num_classes)).unsqueeze(1))  # (31,1)
        num_samples = feature.size(0) # 498
        #num_samples = feature.__len__()
        # y_hat = [hat for head, hat in zip(is_heads, feature) if head == 1]
        # num_split = num_samples
        num_split = ceil(1.0 * num_samples / self.max_len)  # 1
        # feature_use = feature.re
        while True:
            self.clustering_stop(centers)
            if centers is not None:
                self.centers = centers
            if self.stop or loop==5: break

            centers = 0
            count = 0

            start = 0
            for N in range(num_split):
                cur_len = min(self.max_len, num_samples - start)  # 498
                cur_feature = feature.narrow(0, start, cur_len)  # (498,2049)
                cur_feature=torch.reshape(cur_feature,(-1,768))
                dist2center, labels = self.assign_labels(cur_feature)  # dist2center(498,31),labels:(498)
                labels_onehot = to_onehot(labels, self.num_classes)  # (498,31)
                count += torch.sum(labels_onehot, dim=0)  # (31)
                labels = labels.unsqueeze(0)  # (1,498)
                mask = (labels == refs).unsqueeze(2).type(torch.cuda.FloatTensor)  # mask(31,498,1) true和false的矩阵
                reshaped_feature = cur_feature.unsqueeze(0)  # (1, 498, 2048)
                # update centers
                centers += torch.sum(reshaped_feature * mask, dim=1)  # (31,2048)
                start += cur_len  # 498

            mask = (count.unsqueeze(1) > 0).type(torch.cuda.FloatTensor)  # (31,1)
            centers = mask * centers + (1 - mask) * self.init_centers  # (31,2048)
            loop+=1
        dist2center, labels = [], []
        start = 0
        count = 0
        for N in range(num_split):
            cur_len = min(self.max_len, num_samples - start)  # 498
            cur_feature = feature.narrow(0, start, cur_len)  # (498,2048)
            cur_feature = torch.reshape(cur_feature, (-1, 768))
            cur_dist2center, cur_labels = self.assign_labels(cur_feature)  # cur_dist2center(498,31) cur_labels(498)

            # labels_onehot = to_onehot(cur_labels, self.num_classes)  # (498,31)
            # count += torch.sum(labels_onehot, dim=0)  # (31)

            dist2center += [cur_dist2center]  # dist2center[0].size()(498,31)
            labels += [cur_labels]  # labels[0].size()(498)
            start += cur_len  # 498

        self.samples['labels'] = torch.cat(labels, dim=0)
        self.samples['dist2center'] = torch.cat(dist2center, dim=0)

        cluster2label = self.align_centers()  # 31
        # reorder the centers
        self.centers = self.centers[cluster2label, :]  # (31,2048)
        # re-label the data according to the index
        cur_label = []
        self.samples['label'] = []
        num_samples = len(self.samples['feature'])  # 498
        for k in range(num_samples):
            for i in range(len(self.samples['feature'][k])):
                cur_label.append(cluster2label[self.samples['labels'][
                    k * len(self.samples['feature'][k]) + i]].item())  # cluster2label是0-30的数列
            self.samples['label'] += [cur_label]
            cur_label = []

        self.center_change = torch.mean(self.Dist.get_dist(self.centers, \
                                                           self.init_centers))  # self.center_change Out[56]: tensor(0.0542, device='cuda:0')

        for i in range(num_samples):
            # for j in range(len(self.samples['label'][i])):
            self.path2label += [self.samples['label'][i]]
        # for i in range(num_samples):
        #     self.path2label[self.samples['data'][i]] = self.samples['label'][i].item()

        del self.samples['feature']

