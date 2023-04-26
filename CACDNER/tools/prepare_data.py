import os
from solver.base_solver import BaseSolver
import data.utils as data_utils
from data.custom_dataset_dataloader import CustomDatasetDataLoader
from data.class_aware_dataset_dataloader import ClassAwareDataLoader
from data.customwithoutLabel_dataloader import CustomWithoutLabelDatasetDataLoader
from data.train_dataset_dataloader import TrainDatasetDataLoader,TrainTargetDatasetDataLoader
from data.train_comput_dataset_dataloader import TrainComputDatasetDataLoader



from config.config import cfg

def prepare_data():
    dataloaders = {}
    train_transform = data_utils.get_transform(True)
    test_transform = data_utils.get_transform(False)

    source = cfg.DATASET.SOURCE_NAME
    target = cfg.DATASET.TARGET_NAME
    s_dev =  cfg.DATASET.DEV_SOURCE_NAME
    t_dev =  cfg.DATASET.DEV_TARGET_NAME
    dataroot_S = os.path.join(cfg.DATASET.DATAROOT, source)
    dataroot_T = os.path.join(cfg.DATASET.DATAROOT, target)

    with open(os.path.join(cfg.DATASET.DATAROOT, 'category.txt'), 'r') as f:
        classes = f.readlines()
        classes = [c.strip() for c in classes]
    assert (len(classes) == cfg.DATASET.NUM_CLASSES)

    # for clustering
    batch_size = cfg.CLUSTERING.SOURCE_BATCH_SIZE
    dataset_type = 'SingleDataset'
    print('Building clustering_%s dataloader...' % source)
    dataloaders['clustering_' + source] = CustomDatasetDataLoader(
        dataset_root=dataroot_S, dataset_type=dataset_type,
        batch_size=batch_size, transform=train_transform,
        train=False, num_workers=cfg.NUM_WORKERS,
        classnames=classes)

    batch_size = cfg.CLUSTERING.TARGET_BATCH_SIZE
    dataset_type = cfg.CLUSTERING.TARGET_DATASET_TYPE
    print('Building clustering_%s dataloader...' % target)
    dataloaders['clustering_' + target] = CustomWithoutLabelDatasetDataLoader(
        dataset_root=dataroot_T, dataset_type=dataset_type,
        batch_size=batch_size, transform=train_transform,
        train=False, num_workers=cfg.NUM_WORKERS,
        classnames=classes)

    source_dev = cfg.DATASET.SOURCE_NAME+'_dev'
    target_dev = cfg.DATASET.TARGET_NAME+'_dev'
    dataroot_S_dev = os.path.join(cfg.DATASET.DATAROOT, s_dev)
    dataroot_T_dev = os.path.join(cfg.DATASET.DATAROOT, t_dev)
    # class-agnostic source dataloader for supervised training
    batch_size = cfg.TRAIN.SOURCE_BATCH_SIZE
    dataset_type = 'SingleDataset'
    print('Building %s dataloader...' % source_dev)
    dataloaders[source_dev] = TrainComputDatasetDataLoader(
        dataset_root=dataroot_S_dev, dataset_type=dataset_type,
        batch_size=batch_size, transform=train_transform,
        train=True, num_workers=cfg.NUM_WORKERS,
        classnames=classes)

    # class-agnostic source dataloader for supervised training
    batch_size = cfg.TRAIN.TARGET_BATCH_SIZE
    dataset_type = 'SingleDataset'
    print('Building %s dataloader...' % target_dev)
    dataloaders[target_dev] = TrainComputDatasetDataLoader(
        dataset_root=dataroot_T_dev, dataset_type=dataset_type,
        batch_size=batch_size, transform=train_transform,
        train=True, num_workers=cfg.NUM_WORKERS,
        classnames=classes)

    # class-agnostic source dataloader for supervised training
    batch_size = cfg.TRAIN.SOURCE_BATCH_SIZE
    dataset_type = 'SingleDataset'
    data_train_S = os.path.join(cfg.DATASET.DATAROOT,cfg.DATASET.TRAIN_SOURCE_NAME)
    print('Building %s dataloader...' % source)
    dataloaders[source] = TrainDatasetDataLoader(
        dataset_root=dataroot_S, dataset_type=dataset_type,
        batch_size=batch_size, transform=train_transform,
        train=True, num_workers=cfg.NUM_WORKERS,
        classnames=classes)

    # class-agnostic source dataloader for supervised training
    batch_size = cfg.TRAIN.TARGET_BATCH_SIZE
    dataset_type = 'SingleDataset'
    print('Building %s dataloader...' % target)
    dataloaders[target] = TrainTargetDatasetDataLoader(
        dataset_root=dataroot_T, dataset_type=dataset_type,
        batch_size=batch_size, transform=train_transform,
        train=True, num_workers=cfg.NUM_WORKERS,
        classnames=classes)

    # initialize the categorical dataloader
    dataset_type = 'CategoricalSTDataset'
    source_batch_size = cfg.TRAIN.SOURCE_CLASS_BATCH_SIZE
    target_batch_size = cfg.TRAIN.TARGET_CLASS_BATCH_SIZE
    print('Building categorical dataloader...')
    dataloaders['categorical'] = ClassAwareDataLoader(
        dataset_type=dataset_type,
        source_batch_size=source_batch_size,
        target_batch_size=target_batch_size,
        source_dataset_root=dataroot_S,
        transform=train_transform,
        classnames=classes,
        num_workers=cfg.NUM_WORKERS,
        drop_last=True, sampler='RandomSampler')

    batch_size = cfg.TEST.BATCH_SIZE
    dataset_type = cfg.TEST.DATASET_TYPE
    # test_source = cfg.TEST.DOMAIN if cfg.TEST.DOMAIN != "" else target
    test_source = cfg.TEST.SOURCE
    test_target = cfg.TEST.TARGET
    dataroot_test_source = os.path.join(cfg.DATASET.DATAROOT, test_source)
    dataloaders['test_source'] = CustomDatasetDataLoader(
        dataset_root=dataroot_test_source, dataset_type=dataset_type,
        batch_size=batch_size, transform=test_transform,
        train=False, num_workers=cfg.NUM_WORKERS,
        classnames=classes)
    dataroot_test_target = os.path.join(cfg.DATASET.DATAROOT, test_target)
    dataloaders['test_target'] = CustomDatasetDataLoader(
        dataset_root=dataroot_test_target, dataset_type=dataset_type,
        batch_size=batch_size, transform=test_transform,
        train=False, num_workers=cfg.NUM_WORKERS,
        classnames=classes)

    return dataloaders