"""RAF-DB data loading for the archived 92.65% configuration."""

import numpy as np
import torch
from dataloader.rafdb import rafdb


def set_up_datasets(args):
    if args.dataset != "rafdb" or args.classes != 7:
        raise ValueError("This release supports RAF-DB seven-class recognition only.")
    args.Dataset = rafdb
    return args


def get_dataloader(args):
    class_index = np.arange(args.classes)
    trainset = args.Dataset.RafDB(
        root=args.data_path, train=True, index=class_index, args=args
    )
    testset = args.Dataset.RafDB(
        root=args.data_path, train=False, index=class_index, args=args
    )
    print("trainset size: ", len(trainset))
    print("testset size: ", len(testset))
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    return trainset, testset, trainloader, testloader


def get_labelname(args):
    args.label_nms = [
        "surprise", "fear", "disgusted", "happy", "sad", "anger", "neutral"
    ]
    print(args.dataset, "dataset, label_nms: ", args.label_nms)
    return args
