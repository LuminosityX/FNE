import os
import time
import shutil
from data import get_loaders
import ruamel.yaml as yaml

import torch

from model import FNE
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data

import logging
import random
import numpy as np

import argparse

def set_seed(seed = 3407):
    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True



def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_root', default='/data/hdd1/lihaoxuan/originaldata/f30k/images/',
                        help='path to orignal image')
    parser.add_argument('--dataset', default='flickr',
                        help='Dataset: flickr/mscoco')
    
    parser.add_argument('--image_res', default=384, type=int,
                        help='Res of orignal image.')
    parser.add_argument('--batch_size_train', default=32, type=int,                                   
                        help='Size of a training mini-batch.')
    parser.add_argument('--batch_size_test', default=8, type=int,                                   
                        help='Size of a testing mini-batch.')
    
    parser.add_argument('--test', action='store_true',
                        help='Use test mode.')

    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=25, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--learning_rate', default=2e-5, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=15, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=2, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=500, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='runs/runX',
                        help='Path to save the model and Tensorboard log.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')
    
    parser.add_argument('--queue_size', default=8224, type=int,
                        help='Number of memory queue.')
    parser.add_argument('--momentum', default=0.995, type=float,
                        help='Momentum parameter.')
    parser.add_argument('--nums_right_sims', default=200, type=int,
                        help='number of average sims can use.')
    parser.add_argument('--val_times', default=3, type=int,
                        help='Number of times that you want to val.')

    opt = parser.parse_args()

    if opt.dataset == 'flickr':
        opt.train_file = 'json/flickr30k_train.json'
        opt.val_file = 'json/flickr30k_val.json'
        opt.test_file = 'json/flickr30k_test.json'
    else:
        opt.train_file = 'json/coco_train.json'
        opt.val_file = 'json/coco_val.json'
        opt.test_file = 'json/coco_test.json'


    print(opt)

    os.environ['CUDA_VISIBLE_DEVICES']='6'
    set_seed()

    print(os.environ['CUDA_VISIBLE_DEVICES'])

    lr_schedules = [5, 15]

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    

    # Load data loaders
    if not os.path.exists(opt.logger_name):
        os.makedirs(opt.logger_name)

    train_loader, val_loader, test_loader = get_loaders(opt)

    opt.val_step = len(train_loader) // opt.val_times
    print(f'len loader: {len(train_loader)}')
    print(f'real val step: {opt.val_step}')

    # Construct the model
    model = FNE(opt)

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another
            # training
            model.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.resume, start_epoch, best_rsum))
            validate(opt, val_loader, model)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    if opt.test:
        validate(opt, test_loader, model)
        return 
    

def validate(opt, val_loader, model):
    # compute the encoding for all the validation images and captions
    img_embs, cap_embs = encode_data(
        model, val_loader, opt.log_step, logging.info)

    print(img_embs.shape)
    print(cap_embs.shape)

    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, meanr))
    # image retrieval
    (r1i, r5i, r10i, medri, meanri) = t2i(
        img_embs, cap_embs)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanri))
    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r10 + r1i + r5i + r10i

    return currscore