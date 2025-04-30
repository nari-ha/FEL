import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler
from .market1501 import Market1501
from .msmt17 import MSMT17
from .dev_market import DevMarket
from .dev_msmt import DevMSMT
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
from .veri import VeRi

__factory = {
    'market1501': Market1501,
    'msmt17': MSMT17,
    'veri': VeRi,
}

def train_collate_fn(batch):
    imgs, pids, camids, viewids , _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids,

def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths

def make_dataloader(cfg):
    dataset_name = cfg.DATASETS.NAMES
    eval_name = cfg.DATASETS.EVAL
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    if cfg.DEV_MODE == True:
        dataset1 = DevMarket(root=cfg.DATASETS.ROOT_DIR)
        dataset2 = DevMSMT(root=cfg.DATASETS.ROOT_DIR)
    else:
        dataset1 = Market1501(root=cfg.DATASETS.ROOT_DIR)
        dataset2 = MSMT17(root=cfg.DATASETS.ROOT_DIR)
    
    dataset3 = VeRi(root=cfg.DATASETS.ROOT_DIR)
    
    if dataset_name == "person":
        num_classes = dataset1.num_train_pids + dataset2.num_train_pids
        cam_num = dataset1.num_train_cams + dataset2.num_train_cams
        view_num = max(dataset1.num_train_vids, dataset2.num_train_vids)
        train_data = dataset1.train + dataset2.train
    elif dataset_name == "veri":
        num_classes = dataset3.num_train_pids
        cam_num = dataset3.num_train_cams
        view_num = dataset3.num_train_vids
        train_data = dataset3.train
    elif dataset_name == "multi":
        num_classes = dataset1.num_train_pids + dataset2.num_train_pids + dataset3.num_train_pids
        cam_num = dataset1.num_train_cams + dataset2.num_train_cams + dataset3.num_train_cams
        view_num = max(dataset1.num_train_vids, dataset2.num_train_vids, dataset3.num_train_vids)
        train_data = dataset1.train + dataset2.train + dataset3.train
    elif dataset_name == "market1501":
        num_classes = dataset1.num_train_pids
        cam_num = dataset1.num_train_cams
        view_num = dataset1.num_train_vids
        train_data = dataset1.train
    elif dataset_name == "msmt17":
        num_classes = dataset2.num_train_pids
        cam_num = dataset2.num_train_cams
        view_num = dataset2.num_train_vids
        train_data = dataset2.train
    elif dataset_name == "cross":
        # if cfg.DEV_MODE == True:
        num_classes = dataset2.num_train_pids + dataset3.num_train_pids
        cam_num = dataset2.num_train_cams + dataset3.num_train_cams
        view_num = max(dataset2.num_train_vids, dataset3.num_train_vids)
        train_data = dataset2.train + dataset3.train
    
    train_set = ImageDataset(train_data, train_transforms)
    train_set_normal = ImageDataset(train_data, val_transforms)
    
    if eval_name == "market1501":
        query_data = dataset1.query
        gallery_data = dataset1.gallery
    elif eval_name == "msmt17":
        query_data = dataset2.query
        gallery_data = dataset2.gallery
    elif eval_name == "veri":
        query_data = dataset3.query
        gallery_data = dataset3.gallery
    elif eval_name == "person":
        query_data = dataset1.query + dataset2.query
        gallery_data = dataset1.gallery + dataset2.gallery
    elif eval_name == "cross":
        query_data = dataset2.query + dataset3.query
        gallery_data = dataset2.gallery + dataset3.gallery
    elif eval_name == "multi":
        query_data = dataset1.query + dataset2.query + dataset3.query
        gallery_data = dataset1.gallery + dataset2.gallery + dataset3.gallery
        
    val_set = ImageDataset(query_data + gallery_data, val_transforms)
        
    train_loader = DataLoader(
        train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
        sampler=RandomIdentitySampler(train_data, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
        num_workers=num_workers, collate_fn=train_collate_fn
    )
        
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    
    train_loader_normal = DataLoader(
        train_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, train_loader_normal, val_loader, len(query_data), num_classes, cam_num, view_num
