import torch.utils.data
import torch.utils.data._utils.collate as collate



def create_dataloader(dataset, dataset_opt, collate_fn=None):
    if collate_fn is None:
        collate_fn = collate.default_collate
    phase = dataset_opt['phase']
    if 'train' in phase:
        batch_size = dataset_opt['batch_size']
        shuffle = True
        num_workers = dataset_opt['n_workers']
    else:
        batch_size = 1
        shuffle = False
        num_workers = 1

    return torch.utils.data.DataLoader(\
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

def create_dataset(dataset_opt):
    mode = dataset_opt['mode'].upper()
    if mode == 'LR':
        from data.LR_dataset import LRDataset as D
    elif mode == 'LRHR':
        from data.LRHR_dataset import LRHRDataset as D
    elif mode == 'LRHRLRPAN':
        from data.LRHRLRPAN_dataset import LRHRDataset as D
    elif mode=='MSPAN':
        from data.MSPAN_dataset import LRDataset as D
    elif mode=='MSPANLRPAN':
        from data.MSPANLRPAN_dataset import LRDataset as D
    else:
        raise NotImplementedError("Dataset [%s] is not recognized." % mode)

    dataset = D(dataset_opt)
    print('===> [%s] Dataset is created.' % (mode))
    return dataset