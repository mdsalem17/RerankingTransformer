51/1:
from copy import deepcopy
from functools import partial
from pprint import pprint
import os.path as osp
51/2:
import sacred
import torch
import torch.nn as nn
from sacred import SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds
from torch.backends import cudnn
51/3:
from models.ingredient import model_ingredient, get_model
from utils import pickle_load
from utils.data.dataset_ingredient import data_ingredient, get_loaders
# from utils.training import evaluate_time as evaluate
from utils.training import evaluate
51/4: ex = sacred.Experiment('RRT Evaluation', ingredients=[data_ingredient, model_ingredient], interactive=True)
51/5:
# Filter backspaces and linefeeds
SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds
51/6:
cpu = False  # Force training on CPU
cudnn_flag = 'benchmark'
temp_dir = osp.join('logs', 'temp')
resume = None
seed = 0
51/7:  device = torch.device('cuda:0' if torch.cuda.is_available() and not cpu else 'cpu')
51/8: torch.manual_seed(seed)
51/9:
def read_file(filename):
    with open(filename) as f:
        lines = f.read().splitlines()
    return lines
51/10:
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
51/11: (train_set, query_train_set), (query_set, gallery_set) = get_sets('r50_gldv1', '/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/viquae_images','/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/viquae_images','test_query.txt', ('test_query.txt', 'test_gallery.txt'), None, 500)
51/12: from utils.data.dataset import FeatureDataset
51/13: (train_set, query_train_set), (query_set, gallery_set) = get_sets('r50_gldv1', '/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/viquae_images','/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/viquae_images','test_query.txt', ('test_query.txt', 'test_gallery.txt'), None, 500)
51/14: from torch.utils.data import DataLoader, RandomSampler, BatchSampler
51/15:
batch_size      = 36
test_batch_size = 36
max_sequence_len = 500
sampler = 'random'
51/16:
if sampler == 'random':
   train_sampler = BatchSampler(RandomSampler(train_set), batch_size=batch_size, drop_last=False)
51/17:
num_workers = 8  # number of workers used ot load the data
pin_memory  = True  # use the pin_memory option of DataLoader 
recalls = [1, 5, 10]
51/18: num_candidates = 100
51/19:
train_loader = DataLoader(train_set, batch_sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)
query_train_loader = DataLoader(query_train_set, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory)
query_loader   = DataLoader(query_set, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory)
gallery_loader = DataLoader(gallery_set, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory)
51/20:
class MetricLoaders(NamedTuple):
    train: DataLoader
    num_classes: int
    query: DataLoader
    query_train: DataLoader
    gallery: Optional[DataLoader] = None
51/21: from typing import NamedTuple, Optional
51/22:
class MetricLoaders(NamedTuple):
    train: DataLoader
    num_classes: int
    query: DataLoader
    query_train: DataLoader
    gallery: Optional[DataLoader] = None
51/23: loaders, recall_ks = MetricLoaders(train=train_loader, query_train=query_train_loader, query=query_loader, gallery=gallery_loader, num_classes=len(train_set.categories)), recalls
51/24: torch.manual_seed(seed)
51/25:
from models.matcher import MatchERT
from sacred import Ingredient
model_ingredient = Ingredient('model', interactive=True)
51/26:
def get_model(num_global_features, num_local_features, seq_len, dim_K, dim_feedforward, nhead, num_encoder_layers, dropout, activation, normalize_before):
    return MatchERT(d_global=num_global_features, d_model=num_local_features, seq_len=seq_len, d_K=dim_K, nhead=nhead, num_encoder_layers=num_encoder_layers, 
            dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, normalize_before=normalize_before)
51/27:
name = None  
num_global_features = 2048  
num_local_features = 128  
seq_len = None  
dim_K = None  
dim_feedforward = None  
nhead = None  
num_encoder_layers = None  
dropout = 0.0  
activation = "relu"  
normalize_before = False
51/28:
name = 'rrt'
seq_len = 1004
dim_K = 256
dim_feedforward = 1024
nhead = 4
num_encoder_layers = 6
dropout = 0.0 
activation = "relu"
normalize_before = False
51/29: model = get_model(num_global_features,num_local_features,seq_len,dim_K,dim_feedforward,nhead,num_encoder_layers,dropout,activation,normalize_before)
50/98: t = torch.cuda.get_device_properties(0).total_memory
50/99: r = torch.cuda.memory_reserved(0)
50/100: a = torch.cuda.memory_allocated(0)
50/101: f = r-a
50/102: f
50/103:
from pynvml import *
nvmlInit()
h = nvmlDeviceGetHandleByIndex(0)
info = nvmlDeviceGetMemoryInfo(h)
print(f'total    : {info.total}')
print(f'free     : {info.free}')
print(f'used     : {info.used}')
50/104: f // 1024 ** 2
50/105:
for i in tqdm(range(top_k)):
  nnids = medium_nn_inds[:, i]
  index_global    = gallery_global[nnids]
  index_local     = gallery_local[nnids]
  index_mask      = gallery_mask[nnids]
  index_scales    = gallery_scales[nnids]
  index_positions = gallery_positions[nnids]
  current_scores = model(
  query_global, query_local, query_mask, query_scales, query_positions,
    index_global.to(device),
    index_local.to(device),
    index_mask.to(device),
    index_scales.to(device),
    index_positions.to(device))
  scores.append(current_scores.cpu().data)
50/106: torch.cuda.empty_cache()
50/107: f // 1024 ** 2
50/108: r = torch.cuda.memory_reserved(0)
50/109: a = torch.cuda.memory_allocated(0)
50/110: f = r-a
50/111: f // 1024 ** 2
51/30:
if resume is not None:
   checkpoint = torch.load(resume, map_location=torch.device('cpu'))
   model.load_state_dict(checkpoint['state'], strict=True)
51/31:
model.to(device)
model.eval()
loaders.query.dataset.desc_name
loaders.query.dataset.data_dir
nn_inds_path = osp.join(loaders.query.dataset.data_dir, 'nn_inds_%s.pkl'%loaders.query.dataset.desc_name)
cache_nn_inds = torch.from_numpy(pickle_load(nn_inds_path)).long()
51/32: gnd_file = "/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/viquae_images/gnd_dev_viquae.pkl"
51/33: gnd = pickle_load(gnd_file)
51/34: from tqdm import tqdm
51/35:
device = next(model.parameters()).device 
query_global    = query_global.to(device) 
query_local     = query_local.to(device) 
query_mask      = query_mask.to(device) 
query_scales    = query_scales.to(device) 
query_positions = query_positions.to(device) 
num_samples, top_k = cache_nn_inds.size() 
top_k = min(100, top_k)
50/112:
device = next(model.parameters()).device                                                                            
to_device = lambda x: x.to(device, non_blocking=True)                        
query_global, query_local, query_mask, query_scales, query_positions, query_names = [], [], [], [], [], []          
gallery_global, gallery_local, gallery_mask, gallery_scales, gallery_positions, gallery_names = [], [], [], [], [], []
51/36:
device = next(model.parameters()).device                                                                                         
to_device = lambda x: x.to(device, non_blocking=True)                                                                            
query_global, query_local, query_mask, query_scales, query_positions, query_names = [], [], [], [], [], []                 
gallery_global, gallery_local, gallery_mask, gallery_scales, gallery_positions, gallery_names = [], [], [], [], [], []
51/37:
for entry in tqdm(query_loader, desc='Extracting query features', leave=False, ncols=80): 
     q_global, q_local, q_mask, q_scales, q_positions, _, q_names = entry 
     query_global.append(q_global.cpu()) 
     query_local.append(q_local.cpu()) 
     query_mask.append(q_mask.cpu()) 
     query_scales.append(q_scales.cpu()) 
     query_positions.append(q_positions.cpu()) 
     query_names.extend(list(q_names))
51/38:
gallery_global    = torch.cat(gallery_global, 0) 
gallery_local     = torch.cat(gallery_local, 0) 
gallery_mask      = torch.cat(gallery_mask, 0) 
gallery_scales    = torch.cat(gallery_scales, 0) 
gallery_positions = torch.cat(gallery_positions, 0)
51/39:
query_global    = torch.cat(query_global, 0)
query_local     = torch.cat(query_local, 0)
query_mask      = torch.cat(query_mask, 0)
query_scales    = torch.cat(query_scales, 0)
query_positions = torch.cat(query_positions, 0)
51/40:
for entry in tqdm(gallery_loader, desc='Extracting gallery features', leave=False, ncols=80):
    g_global, g_local, g_mask, g_scales, g_positions, _, g_names = entry
    gallery_global.append(g_global.cpu())
    gallery_local.append(g_local.cpu())
    gallery_mask.append(g_mask.cpu())
    gallery_scales.append(g_scales.cpu())
    gallery_positions.append(g_positions.cpu())
    gallery_names.extend(list(g_names))
51/41:
gallery_global    = torch.cat(gallery_global, 0) 
gallery_local     = torch.cat(gallery_local, 0) 
gallery_mask      = torch.cat(gallery_mask, 0) 
gallery_scales    = torch.cat(gallery_scales, 0) 
gallery_positions = torch.cat(gallery_positions, 0)
51/42: torch.cuda.empty_cache()
51/43:
t = torch.cuda.get_device_properties(0).total_memory
r = torch.cuda.memory_reserved(0)                                                                                                
a = torch.cuda.memory_allocated(0)
51/44: f = (r - a)/t * 100
51/45: f
51/46: f = (r - a)/r
51/47: f = (r - a)/r * 100
51/48: f
51/49:
for i in tqdm(range(top_k)): 
   nnids = medium_nn_inds[:, i] 
   index_global    = gallery_global[nnids] 
   index_local     = gallery_local[nnids] 
   index_mask      = gallery_mask[nnids] 
   index_scales    = gallery_scales[nnids] 
   index_positions = gallery_positions[nnids] 
   current_scores = model( 
   query_global, query_local, query_mask, query_scales, query_positions, 
     index_global.to(device), 
     index_local.to(device), 
     index_mask.to(device), 
     index_scales.to(device), 
     index_positions.to(device)) 
   scores.append(current_scores.cpu().data)
51/50: top_k = min(100, top_k)
51/51: num_samples, top_k = cache_nn_inds.size()
51/52: top_k = min(100, top_k)
51/53: medium_nn_inds = deepcopy(cache_nn_inds.cpu().data.numpy())
51/54:
for i in tqdm(range(top_k)): 
   nnids = medium_nn_inds[:, i] 
   index_global    = gallery_global[nnids] 
   index_local     = gallery_local[nnids] 
   index_mask      = gallery_mask[nnids] 
   index_scales    = gallery_scales[nnids] 
   index_positions = gallery_positions[nnids] 
   current_scores = model( 
   query_global, query_local, query_mask, query_scales, query_positions, 
     index_global.to(device), 
     index_local.to(device), 
     index_mask.to(device), 
     index_scales.to(device), 
     index_positions.to(device)) 
   scores.append(current_scores.cpu().data)
51/55: device = next(model.parameters()).device
51/56:
query_global= query_global.to(device)
query_local = query_local.to(device)
query_mask  = query_mask.to(device)
query_scales= query_scales.to(device)
query_positions = query_positions.to(device)
51/57: model = get_model(num_global_features,num_local_features,seq_len,dim_K,dim_feedforward,nhead,num_encoder_layers,dropout,activation,normalize_before)
51/58:
model.to(device)
model.eval()
cache_nn_inds = torch.from_numpy(pickle_load(nn_inds_path)).long()
51/59: exi
   1: %history -g -f history.py
