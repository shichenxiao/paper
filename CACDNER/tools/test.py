import torch
import argparse
import os
import numpy as np
from torch.backends import cudnn
from model import model
import data.utils as data_utils
from utils.utils import to_cuda, mean_accuracy, accuracy
from solver.base_solver import BaseSolver
from data.custom_dataset_dataloader import CustomDatasetDataLoader
import sys
import pprint
from config.config import cfg, cfg_from_file, cfg_from_list
from data.train_comput_dataset_dataloader import TrainComputDatasetDataLoader
from math import ceil as ceil
from data.single_dataset import idx2tag,tag2idx
from sklearn.metrics import precision_recall_fscore_support
from solver.metrics import Metrics
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train script.')
    parser.add_argument('--weights', dest='weights',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--adapted', dest='adapted_model',
                        action='store_true',
                        help='if the model is adapted on target')
    parser.add_argument('--exp_name', dest='exp_name',
                        help='the experiment name', 
                        default='exp', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def save_preds(paths, preds, save_path, filename='preds.txt'):
    assert(len(paths) == preds.size(0))
    with open(os.path.join(save_path, filename), 'w') as f:
        for i in range(len(paths)):
            line = paths[i] + ' ' + str(preds[i].item()) + '\n'
            f.write(line)

def prepare_data():
    dataloaders = {}
    test_transform = data_utils.get_transform(False)

    # target = cfg.TEST.DOMAIN
    target = cfg.TEST.TARGET
    source =cfg.TEST.SOURCE
    dataroot_T = os.path.join(cfg.DATASET.DATAROOT, target)
    dataroot_S = os.path.join(cfg.DATASET.DATAROOT, source)

    with open(os.path.join(cfg.DATASET.DATAROOT, 'category.txt'), 'r') as f:
        classes = f.readlines()
        classes = [c.strip() for c in classes]
    assert(len(classes) == cfg.DATASET.NUM_CLASSES)

    dataloader = None

    dataset_type = cfg.TEST.DATASET_TYPE
    batch_size = cfg.TEST.BATCH_SIZE
    dataloaders['test_source'] = TrainComputDatasetDataLoader(dataset_root=dataroot_S,
                                                        dataset_type=dataset_type, batch_size=batch_size,
                                                        transform=test_transform, train=False,
                                                        num_workers=cfg.NUM_WORKERS, classnames=classes)
    dataloaders['test_target'] = TrainComputDatasetDataLoader(dataset_root=dataroot_T,
                dataset_type=dataset_type, batch_size=batch_size, 
                transform=test_transform, train=False, 
                num_workers=cfg.NUM_WORKERS, classnames=classes)


    return dataloaders

def init_data(dataloader):
    test_data = {key: dict() for key in dataloader}
    for key in test_data.keys():
        if key not in dataloader:
            continue
        cur_dataloader = dataloader[key]
        test_data[key]['loader'] = cur_dataloader
        test_data[key]['iterator'] = None
    return test_data
def test(args):
    # prepare data
    dataloader = init_data(prepare_data())
    # test_data = init_data(dataloader)
    # initialize model
    # model_state_dict = None
    fx_pretrained = True
    #
    # bn_domain_map = {}
    # if cfg.WEIGHTS != '':
    #     weights_dict = torch.load(cfg.WEIGHTS)
    #     model_state_dict = weights_dict['weights']
    #     bn_domain_map = weights_dict['bn_domain_map']
    #     fx_pretrained = False
    #
    # if args.adapted_model:
    #     num_domains_bn = 2
    # else:
    #     num_domains_bn = 1
    num_domains_bn = 1
    model_path = '../exp/30.pt'
    # model_state_dict=torch.load(model_path)


    # net = model.danet(num_classes=cfg.DATASET.NUM_CLASSES,
    #              state_dict=model_state_dict,
    #              feature_extractor=cfg.MODEL.FEATURE_EXTRACTOR,
    #              fx_pretrained=fx_pretrained,
    #              dropout_ratio=cfg.TRAIN.DROPOUT_RATIO,
    #              fc_hidden_dims=cfg.MODEL.FC_HIDDEN_DIMS,
    #              num_domains_bn=num_domains_bn)
    # net = model.danet(num_classes=cfg.DATASET.NUM_CLASSES,
    #                   pretrained_path=cfg.MODEL.PRETRAINED_PATH,
    #                   frozen=[cfg.TRAIN.STOP_GRAD],
    #                   fx_pretrained=fx_pretrained,
    #                   dropout_prob=cfg.TRAIN.DROPOUT_RATIO,
    #                   fc_hidden_dims=cfg.MODEL.FC_HIDDEN_DIMS,
    #                   num_domains_bn=num_domains_bn)
    net = model.danet(num_classes=cfg.DATASET.NUM_CLASSES,
                      # state_dict=model_state_dict,
                      pretrained_path=cfg.MODEL.PRETRAINED_PATH,
                      frozen=[cfg.TRAIN.STOP_GRAD],
                      fx_pretrained=fx_pretrained,
                      dropout_prob=cfg.TRAIN.DROPOUT_RATIO,
                      fc_hidden_dims=cfg.MODEL.FC_HIDDEN_DIMS,
                      num_domains_bn=num_domains_bn)
    net.load_state_dict(torch.load(model_path))
    net = torch.nn.DataParallel(net)

    if torch.cuda.is_available():
        net.cuda()

    # test 
    # res = {}
    # res['path'], res['preds'], res['gt'], res['probs'] = [], [], [], []
    # net.eval()
    #
    # if cfg.TEST.DOMAIN in bn_domain_map:
    #     domain_id = bn_domain_map[cfg.TEST.DOMAIN]
    # else:
    #     domain_id = 0
    dataloader['test_source']['iterator'] = \
        iter(dataloader['test_source']['loader'])
    dataloader['test_target']['iterator'] = \
        iter(dataloader['test_target']['loader'])
    print(f"source_data:")
    fname1 = cfg.EXP_NAME + '/test_source'
    fname2 = cfg.EXP_NAME + '/test_target'
    model_eval(net, dataloader['test_source']['iterator'])
    print(f"target_data:")
    model_eval(net, dataloader['test_target']['iterator'])
    # with torch.no_grad():
    #     net.module.set_bn_domain(domain_id)
    #     for sample in iter(dataloader):
    #         res['path'] += sample['Path']
    #
    #         if cfg.DATA_TRANSFORM.WITH_FIVE_CROP:
    #             n, ncrop, c, h, w = sample['Img'].size()
    #             sample['Img'] = sample['Img'].view(-1, c, h, w)
    #             img = to_cuda(sample['Img'])
    #             probs = net(img)['probs']
    #             probs = probs.view(n, ncrop, -1).mean(dim=1)
    #         else:
    #             img = to_cuda(sample['Img'])
    #             probs = net(img)['probs']
    #
    #         preds = torch.max(probs, dim=1)[1]
    #         res['preds'] += [preds]
    #         res['probs'] += [probs]
    #
    #         if 'Label' in sample:
    #             label = to_cuda(sample['Label'])
    #             res['gt'] += [label]
    #         print('Processed %d samples.' % len(res['path']))
    #
    #     preds = torch.cat(res['preds'], dim=0)
    #     save_preds(res['path'], preds, cfg.SAVE_DIR)
    #
    #     if 'gt' in res and len(res['gt']) > 0:
    #         gts = torch.cat(res['gt'], dim=0)
    #         probs = torch.cat(res['probs'], dim=0)
    #
    #         assert(cfg.EVAL_METRIC == 'mean_accu' or cfg.EVAL_METRIC == 'accuracy')
    #         if cfg.EVAL_METRIC == "mean_accu":
    #             eval_res = mean_accuracy(probs, gts)
    #             print('Test mean_accu: %.4f' % (eval_res))
    #
    #         elif cfg.EVAL_METRIC == "accuracy":
    #             eval_res = accuracy(probs, gts)
    #             print('Test accuracy: %.4f' % (eval_res))

    print('Finished!')


def model_eval(model, iterator):
    model.eval()

    Words,  Tags, Y, Y_hat = [], [], [], []
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
            y_hat = model.module.predict(x, mask)  # y_hat: (N, T)
            t = []
            tags_temp = []
            for tag in tags:
                t.append(tag.split(' '))
            # tags_temp.append(t)
            Words.extend(words)
            # Is_heads.extend(is_heads)
            Tags.extend(t)
            Y.extend(y.numpy().tolist())

            temps = []
            for lst in y_hat:
                temp = []
                for l in lst:
                    temp.append(idx2tag.get(l))
                temps.append(temp)
            Y_hat.extend(temps)

    Metrics(Tags, Y_hat)
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
    #
    # precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
    #
    # print("precision=%.5f" % precision)
    # print("recall=%.5f" % recall)
    # print("f1=%.5f" % f1)

if __name__ == '__main__':
    cudnn.benchmark = True 
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    if args.weights is not None:
        cfg.WEIGHTS = args.weights
    if args.exp_name is not None:
        cfg.EXP_NAME = args.exp_name 

    print('Using config:')
    pprint.pprint(cfg)

    cfg.SAVE_DIR = os.path.join(cfg.SAVE_DIR, cfg.EXP_NAME)
    print('Output will be saved to %s.' % cfg.SAVE_DIR)

    test(args)
