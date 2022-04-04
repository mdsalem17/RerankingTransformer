from copy import deepcopy
from functools import partial
from pprint import pprint
import os.path as osp
import sacred
import torch
import torch.nn as nn
from sacred import SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds
from torch.backends import cudnn
from models.ingredient import model_ingredient, get_model
from utils import pickle_load
from utils.data.dataset_ingredient import data_ingredient, get_loaders
# from utils.training import evaluate_time as evaluate
from utils.training import evaluate
ex = sacred.Experiment('RRT Evaluation', ingredients=[data_ingredient, model_ingredient], interactive=True)
# Filter backspaces and linefeeds
SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds
cpu = False  # Force training on CPU
cudnn_flag = 'benchmark'
temp_dir = osp.join('logs', 'temp')
resume = None
seed = 0
device = torch.device('cuda:0' if torch.cuda.is_available() and not cpu else 'cpu')
torch.manual_seed(seed)
def read_file(filename):
    with open(filename) as f:
        lines = f.read().splitlines()
    return lines
from utils.data.dataset import FeatureDataset
def get_sets(desc_name, 
        train_data_dir, test_data_dir, 
        train_txt, test_txt, test_gnd_file, 
        max_sequence_len):
    ####################################################################################################################################
    train_lines     = read_file(osp.join(train_data_dir, train_txt))
    train_samples   = [(line.split(';;')[0], int(line.split(';;')[1]), int(line.split(';;')[2]), int(line.split(';;')[3])) for line in train_lines]
    train_set       = FeatureDataset(train_data_dir, train_samples, desc_name, max_sequence_len)
    query_train_set = FeatureDataset(train_data_dir, train_samples, desc_name, max_sequence_len)
    ####################################################################################################################################
    test_gnd_data = None if test_gnd_file is None else pickle_load(osp.join(test_data_dir, test_gnd_file))
    query_lines   = read_file(osp.join(test_data_dir, test_txt[0]))
    gallery_lines = read_file(osp.join(test_data_dir, test_txt[1]))
    query_samples   = [(line.split(';;')[0], int(line.split(';;')[1]), int(line.split(';;')[2]), int(line.split(';;')[3])) for line in query_lines]
    gallery_samples = [(line.split(';;')[0], int(line.split(';;')[1]), int(line.split(';;')[2]), int(line.split(';;')[3])) for line in gallery_lines]
    gallery_set = FeatureDataset(test_data_dir, gallery_samples, desc_name, max_sequence_len)
    query_set   = FeatureDataset(test_data_dir, query_samples,   desc_name, max_sequence_len, gnd_data=test_gnd_data)
        
    return (train_set, query_train_set), (query_set, gallery_set)
(train_set, query_train_set), (query_set, gallery_set) = get_sets('r50_gldv1', '/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/viquae_images','/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/viquae_images','test_query.txt', ('test_query.txt', 'test_gallery.txt'), None, 500)
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
batch_size      = 36
test_batch_size = 36
max_sequence_len = 500
sampler = 'random'
if sampler == 'random':
   train_sampler = BatchSampler(RandomSampler(train_set), batch_size=batch_size, drop_last=False)
num_workers = 8  # number of workers used ot load the data
pin_memory  = True  # use the pin_memory option of DataLoader 
num_candidates = 100
recalls = [1, 5, 10]
train_loader = DataLoader(train_set, batch_sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)
query_train_loader = DataLoader(query_train_set, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory)
query_loader   = DataLoader(query_set, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory)
gallery_loader = DataLoader(gallery_set, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory)
class MetricLoaders(NamedTuple):
    train: DataLoader
    num_classes: int
    query: DataLoader
    query_train: DataLoader
    gallery: Optional[DataLoader] = None
from typing import NamedTuple, Optional
class MetricLoaders(NamedTuple):
    train: DataLoader
    num_classes: int
    query: DataLoader
    query_train: DataLoader
    gallery: Optional[DataLoader] = None
loaders, recall_ks = MetricLoaders(train=train_loader, query_train=query_train_loader, query=query_loader, gallery=gallery_loader, num_classes=len(train_set.categories)), recalls
torch.manual_seed(seed)
from models.matcher import MatchERT
from sacred import Ingredient
model_ingredient = Ingredient('model', interactive=True)
def get_model(num_global_features, num_local_features, seq_len, dim_K, dim_feedforward, nhead, num_encoder_layers, dropout, activation, normalize_before):
    return MatchERT(d_global=num_global_features, d_model=num_local_features, seq_len=seq_len, d_K=dim_K, nhead=nhead, num_encoder_layers=num_encoder_layers, 
            dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, normalize_before=normalize_before)
name = 'rrt'
num_global_features = 2048  
num_local_features = 128  
seq_len = 1004
dim_K = 256
dim_feedforward = 1024
nhead = 4
num_encoder_layers = 6
dropout = 0.0 
activation = "relu"
normalize_before = False
model = get_model(num_global_features,num_local_features,seq_len,dim_K,dim_feedforward,nhead,num_encoder_layers,dropout,activation,normalize_before)
if resume is not None:
   checkpoint = torch.load(resume, map_location=torch.device('cpu'))
   model.load_state_dict(checkpoint['state'], strict=True)
model.to(device)
model.eval() 	
loaders.query.dataset.desc_name
loaders.query.dataset.desc_name
loaders.query.dataset.data_dir
nn_inds_path = osp.join(loaders.query.dataset.data_dir, 'nn_inds_%s.pkl'%loaders.query.dataset.desc_name)
cache_nn_inds = torch.from_numpy(pickle_load(nn_inds_path)).long()
eval_function = partial(evaluate, model=model, 
        cache_nn_inds=cache_nn_inds,
        recall=recall_ks, query_loader=loaders.query, gallery_loader=loaders.gallery)
%history -f swing.py
