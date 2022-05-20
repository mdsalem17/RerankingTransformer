import os.path as osp
from copy import deepcopy
from sacred import Ingredient
from typing import NamedTuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, BatchSampler

from .dataset import FeatureDataset
from .utils import pickle_load, TripletSampler


data_ingredient = Ingredient('dataset')


@data_ingredient.config
def config():
    name = None
    set_name = ''
    eval_set_name = ''
    desc_name = None
    train_data_dir = None
    test_data_dir  = None
    train_txt = None
    test_txt  = None
    train_gnd_file = None
    test_gnd_file = None
    
    batch_size      = 36
    test_batch_size = 36
    max_sequence_len = 500
    split_char  = ','
    sampler = 'random'
    prefixed = None

    num_workers = 8  # number of workers used to load the data
    pin_memory  = True  # use the pin_memory option of DataLoader 
    recalls = [1, 5, 10]
    ###############################################
    ## Negative sampling
    num_candidates = 100

################################################################################################################
### GLDV1

@data_ingredient.named_config
def viquae_tuto_r50_gldv1():
    name = 'viquae_tuto_r50_gldv1'
    set_name = 'tuto'
    eval_set_name = set_name
    train_txt = 'tuto_query.txt'
    test_txt = ('tuto_query.txt', 'tuto_selection.txt')
    train_data_dir = 'data/viquae_for_rrt'
    test_data_dir  = 'data/viquae_for_rrt'
    test_gnd_file = 'gnd_tuto.pkl'
    desc_name = 'r50_gldv1'
    split_char  = ';;'
    sampler = 'random'


@data_ingredient.named_config
def viquae_train_r50_gldv1():
    name = 'viquae_train_r50_gldv1'
    set_name = 'train'
    eval_set_name = set_name
    train_txt = 'train_query.txt'
    test_txt = ('train_query.txt', 'train_selection.txt')
    train_data_dir = 'data/viquae_for_rrt'
    test_data_dir  = 'data/viquae_for_rrt'
    test_gnd_file = 'gnd_train.pkl'
    desc_name = 'r50_gldv1'
    split_char  = ';;'
    sampler = 'random'

@data_ingredient.named_config
def viquae_dev_r50_gldv1():
    name = 'viquae_dev_r50_gldv1'
    set_name = 'dev'
    eval_set_name = set_name
    train_txt = 'dev_query.txt'
    test_txt = ('dev_query.txt', 'dev_selection.txt')
    train_data_dir = 'data/viquae_for_rrt'
    test_data_dir  = 'data/viquae_for_rrt'
    test_gnd_file = 'gnd_dev.pkl'
    desc_name = 'r50_gldv1'
    split_char  = ';;'
    sampler = 'random'
    
@data_ingredient.named_config
def viquae_test_r50_gldv1():
    name = 'viquae_test_r50_gldv1'
    set_name = 'test'
    eval_set_name = set_name
    train_txt = 'test_query.txt'
    test_txt = ('test_query.txt', 'test_selection.txt')
    train_data_dir = 'data/viquae_for_rrt'
    test_data_dir  = 'data/viquae_for_rrt'
    test_gnd_file = 'gnd_test.pkl'
    desc_name = 'r50_gldv1'
    split_char  = ';;'
    sampler = 'random'


################################################################################################################
### GLDV2

@data_ingredient.named_config
def viquae_tuto_r50_gldv2():
    name = 'viquae_tuto_r50_gldv2'
    set_name = 'tuto'
    eval_set_name = set_name
    train_txt = 'tuto_query.txt'
    test_txt = ('tuto_query.txt', 'tuto_selection.txt')
    train_data_dir = 'data/viquae_for_rrt'
    test_data_dir  = 'data/viquae_for_rrt'
    test_gnd_file = 'gnd_tuto.pkl'
    desc_name = 'r50_gldv2'
    split_char  = ';;'
    sampler = 'random'


@data_ingredient.named_config
def viquae_train_r50_gldv2():
    name = 'viquae_train_r50_gldv2'
    set_name = 'train'
    eval_set_name = set_name
    train_txt = 'train_query.txt'
    test_txt = ('train_query.txt', 'train_selection.txt')
    train_data_dir = 'data/viquae_for_rrt'
    test_data_dir  = 'data/viquae_for_rrt'
    test_gnd_file = 'gnd_train.pkl'
    desc_name = 'r50_gldv2'
    split_char  = ';;'
    sampler = 'random'

@data_ingredient.named_config
def viquae_dev_r50_gldv2():
    name = 'viquae_dev_r50_gldv2'
    set_name = 'dev'
    eval_set_name = set_name
    train_txt = 'dev_query.txt'
    test_txt = ('dev_query.txt', 'dev_selection.txt')
    train_data_dir = 'data/viquae_for_rrt'
    test_data_dir  = 'data/viquae_for_rrt'
    test_gnd_file = 'gnd_dev.pkl'
    desc_name = 'r50_gldv2'
    split_char  = ';;'
    sampler = 'random'
    
@data_ingredient.named_config
def viquae_test_r50_gldv2():
    name = 'viquae_test_r50_gldv2'
    set_name = 'test'
    eval_set_name = set_name
    train_txt = 'test_query.txt'
    test_txt = ('test_query.txt', 'test_selection.txt')
    train_data_dir = 'data/viquae_for_rrt'
    test_data_dir  = 'data/viquae_for_rrt'
    test_gnd_file = 'gnd_test.pkl'
    desc_name = 'r50_gldv2'
    split_char  = ';;'
    sampler = 'random'

################################################################################################################
### Training ViQuAE Resnet50

@data_ingredient.named_config
def train_viquae_dev_r50_gldv1():
    name = 'train_viquae_dev_r50_gldv1'
    set_name = 'train'
    eval_set_name = 'dev'
    train_txt = ('train_query.txt', 'train_gallery.txt')
    test_txt = ('dev_query.txt', 'dev_selection.txt')
    train_data_dir = 'data/viquae_for_rrt'
    test_data_dir  = 'data/viquae_for_rrt'
    train_gnd_file = 'gnd_train.pkl'
    test_gnd_file = 'gnd_dev.pkl'
    desc_name = 'r50_gldv1'
    sampler = 'triplet'
    split_char  = ';;'


@data_ingredient.named_config
def train_viquae_dev_r50_gldv2():
    name = 'train_viquae_dev_r50_gldv2'
    set_name = 'train'
    eval_set_name = 'dev'
    train_txt = ('train_query.txt', 'train_gallery.txt')
    test_txt = ('dev_query.txt', 'dev_selection.txt')
    train_data_dir = 'data/viquae_for_rrt'
    test_data_dir  = 'data/viquae_for_rrt'
    train_gnd_file = 'gnd_train.pkl'
    test_gnd_file = 'gnd_dev.pkl'
    desc_name = 'r50_gldv2'
    sampler = 'triplet'
    split_char  = ';;'


@data_ingredient.named_config
def tuto_viquae_tuto_r50_gldv1():
    name = 'tuto_viquae_tuto_r50_gldv1'
    set_name = 'tuto'
    eval_set_name = 'tuto'
    train_txt = ('tuto_query.txt', 'tuto_gallery.txt')
    test_txt = ('tuto_query.txt', 'tuto_selection.txt')
    train_data_dir = 'data/viquae_for_rrt'
    test_data_dir  = 'data/viquae_for_rrt'
    train_gnd_file = 'gnd_tuto.pkl'
    test_gnd_file = 'gnd_tuto.pkl'
    desc_name = 'r50_gldv1'
    sampler = 'triplet'
    split_char  = ';;'


@data_ingredient.named_config
def tuto_viquae_tuto_r50_gldv2():
    name = 'tuto_viquae_tuto_r50_gldv2'
    set_name = 'tuto'
    eval_set_name = 'tuto'
    train_txt = ('tuto_query.txt', 'tuto_gallery.txt')
    test_txt = ('tuto_query.txt', 'tuto_selection.txt')
    train_data_dir = 'data/viquae_for_rrt'
    test_data_dir  = 'data/viquae_for_rrt'
    train_gnd_file = 'gnd_tuto.pkl'
    test_gnd_file = 'gnd_tuto.pkl'
    desc_name = 'r50_gldv2'
    sampler = 'triplet'
    split_char  = ';;'


################################################################################################################
### Revisited Oxford Resnet50
    
@data_ingredient.named_config
def roxford_r50_gldv1():
    name = 'roxford_r50_gldv1'
    train_txt = 'test_query.txt'
    test_txt = ('test_query.txt', 'test_gallery.txt')
    train_data_dir = 'data/oxford5k'
    test_data_dir  = 'data/oxford5k'
    test_gnd_file = 'gnd_roxford5k.pkl'
    desc_name = 'r50_gldv1'
    sampler = 'random'


@data_ingredient.named_config
def roxford_r50_gldv2():
    name = 'roxford_r50_gldv2'
    train_txt = 'test_query.txt'
    test_txt = ('test_query.txt', 'test_gallery.txt')
    train_data_dir = 'data/oxford5k'
    test_data_dir  = 'data/oxford5k'
    test_gnd_file = 'gnd_roxford5k.pkl'
    desc_name = 'r50_gldv2'
    sampler = 'random'

################################################################################################################
### Revisited Paris Resnet50

@data_ingredient.named_config
def rparis_r50_gldv1():
    name = 'rparis_r50_gldv1'
    train_txt = 'test_query.txt'
    test_txt = ('test_query.txt', 'test_gallery.txt')
    train_data_dir = 'data/paris6k'
    test_data_dir  = 'data/paris6k'
    test_gnd_file = 'gnd_rparis6k.pkl'
    desc_name = 'r50_gldv1'
    sampler = 'random'


@data_ingredient.named_config
def rparis_r50_gldv2():
    name = 'rparis_r50_gldv2'
    train_txt = 'test_query.txt'
    test_txt = ('test_query.txt', 'test_gallery.txt')
    train_data_dir = 'data/paris6k'
    test_data_dir  = 'data/paris6k'
    test_gnd_file = 'gnd_rparis6k.pkl'
    desc_name = 'r50_gldv2'
    sampler = 'random'

################################################################################################################
### Revisited Oxford Resnet101

@data_ingredient.named_config
def roxford_r101_gldv1():
    name = 'roxford_r101_gldv1'
    train_txt = 'test_query.txt'
    test_txt = ('test_query.txt', 'test_gallery.txt')
    train_data_dir = 'data/oxford5k'
    test_data_dir  = 'data/oxford5k'
    test_gnd_file = 'gnd_roxford5k.pkl'
    desc_name = 'r101_gldv1'
    sampler = 'random'


@data_ingredient.named_config
def roxford_r101_gldv2():
    name = 'roxford_r101_gldv2'
    train_txt = 'test_query.txt'
    test_txt = ('test_query.txt', 'test_gallery.txt')
    train_data_dir = 'data/oxford5k'
    test_data_dir  = 'data/oxford5k'
    test_gnd_file = 'gnd_roxford5k.pkl'
    desc_name = 'r101_gldv2'
    sampler = 'random'

################################################################################################################
### Revisited Paris Resnet101

@data_ingredient.named_config
def rparis_r101_gldv1():
    name = 'rparis_r101_gldv1'
    train_txt = 'test_query.txt'
    test_txt = ('test_query.txt', 'test_gallery.txt')
    train_data_dir = 'data/paris6k'
    test_data_dir  = 'data/paris6k'
    test_gnd_file = 'gnd_rparis6k.pkl'
    desc_name = 'r101_gldv1'
    sampler = 'random'


@data_ingredient.named_config
def rparis_r101_gldv2():
    name = 'rparis_r101_gldv2'
    train_txt = 'test_query.txt'
    test_txt = ('test_query.txt', 'test_gallery.txt')
    train_data_dir = 'data/paris6k'
    test_data_dir  = 'data/paris6k'
    test_gnd_file = 'gnd_rparis6k.pkl'
    desc_name = 'r101_gldv2'
    sampler = 'random'


################################################################################################################
### Training RRT on GLDv2

@data_ingredient.named_config
def gldv2_roxford_r50_gldv1():
    name = 'gldv2_roxford_r50_gldv1'
    train_txt = 'train.txt'
    test_txt = ('test_query.txt', 'test_gallery.txt')
    train_data_dir = 'data/gldv2'
    test_data_dir  = 'data/oxford5k'
    test_gnd_file = 'gnd_roxford5k.pkl'
    desc_name = 'r50_gldv1'
    sampler = 'triplet'


@data_ingredient.named_config
def gldv2_roxford_r101_gldv1():
    name = 'gldv2_roxford_r101_gldv1'
    train_txt = 'train.txt'
    test_txt = ('test_query.txt', 'test_gallery.txt')
    train_data_dir = 'data/gldv2'
    test_data_dir  = 'data/oxford5k'
    test_gnd_file = 'gnd_roxford5k.pkl'
    desc_name = 'r101_gldv1'
    sampler = 'triplet'


@data_ingredient.named_config
def gldv2_roxford_r50_gldv2():
    name = 'gldv2_roxford_r50_gldv2'
    train_txt = 'train.txt'
    test_txt = ('test_query.txt', 'test_gallery.txt')
    train_data_dir = 'data/gldv2'
    test_data_dir  = 'data/oxford5k'
    test_gnd_file = 'gnd_roxford5k.pkl'
    desc_name = 'r50_gldv2'
    sampler = 'triplet'


@data_ingredient.named_config
def gldv2_roxford_r101_gldv2():
    name = 'gldv2_roxford_r101_gldv2'
    train_txt = 'train.txt'
    test_txt = ('test_query.txt', 'test_gallery.txt')
    train_data_dir = 'data/gldv2'
    test_data_dir  = 'data/oxford5k'
    test_gnd_file = 'gnd_roxford5k.pkl'
    desc_name = 'r101_gldv2'
    sampler = 'triplet'

################################################################################################################


def read_file(filename):
    with open(filename) as f:
        lines = f.read().splitlines()
    return lines


class MetricLoaders(NamedTuple):
    train: DataLoader
    num_classes: int
    query: DataLoader
    query_train: DataLoader
    prefixed: str = None
    set_name: str = ''
    eval_set_name: str = ''
    gallery: Optional[DataLoader] = None


@data_ingredient.capture
def get_sets(desc_name, 
        train_data_dir, test_data_dir, train_txt, 
        test_txt, train_gnd_file,  test_gnd_file,
        max_sequence_len, split_char, prefixed):
    ####################################################################################################################################
    train_gnd_file = prefixed+'_'+train_gnd_file if prefixed is not None and train_gnd_file is not None else train_gnd_file
    test_gnd_file  = prefixed+'_'+test_gnd_file  if prefixed is not None and test_gnd_file  is not None else test_gnd_file
    
    if len(train_txt) == 2:
        
        train_gnd_data  = None if train_gnd_file is None else pickle_load(osp.join(train_data_dir, train_gnd_file))
        train_lines_txt = train_txt[1] if prefixed is None else prefixed+'_'+train_txt[1]
        train_lines     = read_file(osp.join(train_data_dir, train_lines_txt))
        train_q_lines_txt = train_txt[0] if prefixed is None else prefixed+'_'+train_txt[0]
        train_q_lines   = read_file(osp.join(train_data_dir, train_q_lines_txt))
        train_samples   = [(line.split(split_char)[0], int(line.split(split_char)[1]), int(line.split(split_char)[2]), int(line.split(split_char)[3])) for line in train_lines]
        train_q_samples = [(line.split(split_char)[0], int(line.split(split_char)[1]), int(line.split(split_char)[2]), int(line.split(split_char)[3])) for line in train_q_lines]
        train_set       = FeatureDataset(train_data_dir, train_samples,   desc_name, max_sequence_len, gnd_data=train_gnd_data)
        query_train_set = FeatureDataset(train_data_dir, train_q_samples, desc_name, max_sequence_len, gnd_data=train_gnd_data)
    else:
        train_gnd_data  = None if train_gnd_file is None else pickle_load(osp.join(train_data_dir, train_gnd_file))
        train_lines_txt = train_txt if prefixed is None else prefixed+'_'+train_txt
        train_lines     = read_file(osp.join(train_data_dir, train_lines_txt))
        train_samples   = [(line.split(split_char)[0], int(line.split(split_char)[1]), int(line.split(split_char)[2]), int(line.split(split_char)[3])) for line in train_lines]
        train_set       = FeatureDataset(train_data_dir, train_samples, desc_name, max_sequence_len, gnd_data=train_gnd_data)
        query_train_set = FeatureDataset(train_data_dir, train_samples, desc_name, max_sequence_len, gnd_data=train_gnd_data)
        ####################################################################################################################################
    test_gnd_data   = None if test_gnd_file is None else pickle_load(osp.join(test_data_dir, test_gnd_file))
    query_lines_txt = test_txt[0] if prefixed is None else prefixed+'_'+test_txt[0]
    print('query_lines_txt', query_lines_txt)
    query_lines     = read_file(osp.join(test_data_dir, query_lines_txt))
    gallery_lines_txt = test_txt[1] if prefixed is None else prefixed+'_'+test_txt[1]
    gallery_lines   = read_file(osp.join(test_data_dir, gallery_lines_txt))
    query_samples   = [(line.split(split_char)[0], int(line.split(split_char)[1]), int(line.split(split_char)[2]), int(line.split(split_char)[3])) for line in query_lines]
    gallery_samples = [(line.split(split_char)[0], int(line.split(split_char)[1]), int(line.split(split_char)[2]), int(line.split(split_char)[3])) for line in gallery_lines]
    gallery_set     = FeatureDataset(test_data_dir, gallery_samples, desc_name, max_sequence_len)
    query_set       = FeatureDataset(test_data_dir, query_samples,   desc_name, max_sequence_len, gnd_data=test_gnd_data)
        
    return (train_set, query_train_set), (query_set, gallery_set)


@data_ingredient.capture
def get_loaders(desc_name, train_data_dir, 
    batch_size, test_batch_size, 
    num_workers, pin_memory, 
    sampler, recalls, set_name, 
    eval_set_name, train_gnd_file,
    prefixed, num_candidates=100):

    (train_set, query_train_set), (query_set, gallery_set) = get_sets()

    if sampler == 'random':
        train_sampler = BatchSampler(RandomSampler(train_set), batch_size=batch_size, drop_last=False)
    elif sampler == 'triplet':
        nn_inds_path = set_name+'_nn_inds_%s.pkl'%desc_name if prefixed is None else prefixed+'_'+set_name + '_nn_inds_%s.pkl'%desc_name
        train_nn_inds = osp.join(train_data_dir, nn_inds_path)
        gnd_data = train_set.gnd_data['gnd']
        train_sampler = TripletSampler(query_train_set.targets, batch_size, train_nn_inds, num_candidates, gnd_data)
    else:
        raise ValueError('Invalid choice of sampler ({}).'.format(sampler))
    train_loader = DataLoader(train_set, batch_sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)
    query_train_loader = DataLoader(query_train_set, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory)
        
    query_loader   = DataLoader(query_set, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory)
    gallery_loader = DataLoader(gallery_set, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory)

    return MetricLoaders(train=train_loader, query_train=query_train_loader, query=query_loader, gallery=gallery_loader, num_classes=len(train_set.categories),set_name=set_name,eval_set_name=eval_set_name,prefixed=prefixed), recalls
