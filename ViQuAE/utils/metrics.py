import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
from copy import deepcopy
from typing import Dict, List

from .revisited import compute_metrics
from .data.utils import json_save, pickle_save

import time
import gc

def fill_in_and_pad(gallery_in, query, sizes):
    shape = list(query.shape)
    shape.insert(1, 100)
    gallery_out = torch.zeros(shape)
    size = 0
    counter = 0
    for i in range(gallery_out.size(dim=0)):
        for j in range(gallery_out.size(dim=1)):
            if j < sizes[i]:
                gallery_out[i][j] = gallery_in[counter]
                counter += 1
    
    return gallery_out


def remove_padded_indices(rankings, nn_inds, sizes):
    # rankings.shape -> (nb_queries x 100)
    assert(len(rankings) == len(sizes))
    for i in range(len(sizes)):
        assert(max(nn_inds[i, :sizes[i]]) < sizes[i])
        rankings[i, :sizes[i]] = np.array([value for value in rankings[i] if value < sizes[i]])
        rankings[i, sizes[i]:] = nn_inds[i, sizes[i]:]
        
    return rankings



class AverageMeter:
    """Computes and stores the average and current value on device"""

    def __init__(self, device, length):
        self.device = device
        self.length = length
        self.reset()

    def reset(self):
        self.values = torch.zeros(self.length, device=self.device, dtype=torch.float)
        self.counter = 0
        self.last_counter = 0

    def append(self, val):
        self.values[self.counter] = val.detach()
        self.counter += 1
        self.last_counter += 1

    @property
    def val(self):
        return self.values[self.counter - 1]

    @property
    def avg(self):
        return self.values[:self.counter].mean()

    @property
    def values_list(self):
        return self.values[:self.counter].cpu().tolist()

    @property
    def last_avg(self):
        if self.last_counter == 0:
            return self.latest_avg
        else:
            self.latest_avg = self.values[self.counter - self.last_counter:self.counter].mean()
            self.last_counter = 0
            return self.latest_avg


@torch.no_grad()
def mean_average_precision_viquae_rerank(
    model: nn.Module,
    cache_nn_inds: torch.Tensor,
    query_global: torch.Tensor, query_local: torch.Tensor, query_mask: torch.Tensor, query_scales: torch.Tensor, query_positions: torch.Tensor,
    gallery_global: torch.Tensor, gallery_local: torch.Tensor, gallery_mask: torch.Tensor, gallery_scales: torch.Tensor, gallery_positions: torch.Tensor,
    ks: List[int],
    gnd, set_name=None, max_sequence_len=None) -> Dict[str, float]:

    device = next(model.parameters()).device
    query_global    = query_global.to(device)
    query_local     = query_local.to(device)
    query_mask      = query_mask.to(device)
    query_scales    = query_scales.to(device)
    query_positions = query_positions.to(device)

    num_samples, top_k = cache_nn_inds.size()
    top_k = min(100, top_k)
    
    sizes = [len(gnd['simlist'][i]) for i in range(len(gnd['simlist']))]
    
    gallery_global    = fill_in_and_pad(gallery_global, query_global, sizes)
    gallery_local     = fill_in_and_pad(gallery_local, query_local, sizes)
    gallery_mask      = fill_in_and_pad(gallery_mask, query_mask, sizes)
    gallery_scales    = fill_in_and_pad(gallery_scales, query_scales, sizes)
    gallery_positions = fill_in_and_pad(gallery_positions, query_positions, sizes)


    ########################################################################################
    ## Evaluation
    eval_nn_inds = deepcopy(cache_nn_inds.cpu().data.numpy())

    ## Exclude the junk images as in DELG (https://github.com/tensorflow/models/blob/44cad43aadff9dd12b00d4526830f7ea0796c047/research/delf/delf/python/detect_to_retrieve/image_reranking.py#L190)
    #for i in range(num_samples):
    #    junk_ids = gnd['gnd'][i]['r_neg']
    #    all_ids = eval_nn_inds[i]
    #    pos = np.in1d(all_ids, junk_ids)
    #    neg = np.array([not x for x in pos])
    #    new_ids = np.concatenate([np.arange(len(all_ids))[neg], np.arange(len(all_ids))[pos]])
    #    new_ids = all_ids[new_ids]
    #    eval_nn_inds[i] = new_ids
    eval_nn_inds = torch.from_numpy(eval_nn_inds)
    n_queries = gallery_global.size(dim=0)
    
    scores = []
    for i in tqdm(range(n_queries)):
        
        q_global    = query_global[i].unsqueeze(dim=0)
        q_local     = query_local[i].unsqueeze(dim=0)
        q_mask      = query_mask[i].unsqueeze(dim=0)
        q_scales    = query_scales[i].unsqueeze(dim=0)
        q_positions = query_positions[i].unsqueeze(dim=0)
        
        q_scores =  []
        for j in range(top_k):
            index_global = gallery_global[i, j]
            index_local = gallery_local[i, j]
            index_mask = gallery_mask[i, j]
            index_scales = gallery_scales[i, j]
            index_positions = gallery_positions[i, j]
            
            index_global = index_global.unsqueeze(dim=0)
            index_global = index_global.type(torch.float32)

            index_local = index_local.unsqueeze(dim=0)
            index_local = index_local.type(torch.float32)

            index_mask = index_mask.unsqueeze(dim=0)
            index_mask = index_mask.type(torch.bool)

            index_scales = index_scales.unsqueeze(dim=0)
            index_scales = index_scales.type(torch.int64)

            index_positions = index_positions.unsqueeze(dim=0)
            index_positions = index_positions.type(torch.float32)
            


            iter_scores = model(
                q_global, q_local, q_mask, q_scales, q_positions,
                index_global.to(device),
                index_local.to(device),
                index_mask.to(device),
                index_scales.to(device),
                index_positions.to(device))
            
            q_scores.append(iter_scores.cpu().data)
        
        current_scores = torch.from_numpy(np.stack(q_scores, axis=0)).squeeze(1)
        torch.cuda.empty_cache() 
        scores.append(current_scores.cpu().data)
    
    
    scores = torch.stack(scores, axis=0) # nb_queries x 100
    closest_dists, closest_indices = torch.sort(scores, dim=-1, descending=True)
    ranks = deepcopy(eval_nn_inds)
    ranks[:, :top_k] = deepcopy(closest_indices)
    ranks = ranks.cpu().data.numpy()
    ranks = remove_padded_indices(ranks, eval_nn_inds, sizes)
    
    out = compute_metrics('viquae', ranks.T, gnd['gnd'], sizes, kappas=ks, set_name=set_name, max_sequence_len=max_sequence_len)

    ########################################################################################  
    
    return out

@torch.no_grad()
def mean_average_precision_revisited_rerank(
    model: nn.Module,
    cache_nn_inds: torch.Tensor,
    query_global: torch.Tensor, query_local: torch.Tensor, query_mask: torch.Tensor, query_scales: torch.Tensor, query_positions: torch.Tensor,
    gallery_global: torch.Tensor, gallery_local: torch.Tensor, gallery_mask: torch.Tensor, gallery_scales: torch.Tensor, gallery_positions: torch.Tensor,
    ks: List[int],
    gnd) -> Dict[str, float]:

    device = next(model.parameters()).device
    query_global    = query_global.to(device)
    query_local     = query_local.to(device)
    query_mask      = query_mask.to(device)
    query_scales    = query_scales.to(device)
    query_positions = query_positions.to(device)

    num_samples, top_k = cache_nn_inds.size()
    top_k = min(100, top_k)
    
    print("top_k: ", top_k)    
    print("gallery_global size: ", gallery_global.shape)
    print("gallery_local size: ", gallery_local.shape)
    print("gallery_mask size: ", gallery_mask.shape)
    print("gallery_scales size: ", gallery_scales.shape)
    print("gallery_positions size: ", gallery_positions.shape)

    ########################################################################################
    ## Medium
    medium_nn_inds = deepcopy(cache_nn_inds.cpu().data.numpy())

    # Exclude the junk images as in DELG (https://github.com/tensorflow/models/blob/44cad43aadff9dd12b00d4526830f7ea0796c047/research/delf/delf/python/detect_to_retrieve/image_reranking.py#L190)
    for i in range(num_samples):
        junk_ids = gnd['gnd'][i]['junk']
        all_ids = medium_nn_inds[i]
        pos = np.in1d(all_ids, junk_ids)
        neg = np.array([not x for x in pos])
        new_ids = np.concatenate([np.arange(len(all_ids))[neg], np.arange(len(all_ids))[pos]])
        new_ids = all_ids[new_ids]
        medium_nn_inds[i] = new_ids
    medium_nn_inds = torch.from_numpy(medium_nn_inds)
    
    scores = []
    for i in tqdm(range(top_k)):
        nnids = medium_nn_inds[:, i]
        index_global    = gallery_global[nnids]
        index_local     = gallery_local[nnids]
        index_mask      = gallery_mask[nnids]
        index_scales    = gallery_scales[nnids]
        index_positions = gallery_positions[nnids]
        
        print("index_global size: ", index_global.shape)
        print("index_local size: ", index_local.shape)
        print("index_mask size: ", index_mask.shape)
        print("index_scales size: ", index_scales.shape)
        print("index_positions size: ", index_positions.shape)
        current_scores = model(
            query_global, query_local, query_mask, query_scales, query_positions,
            index_global.to(device),
            index_local.to(device),
            index_mask.to(device),
            index_scales.to(device),
            index_positions.to(device))
        scores.append(current_scores.cpu().data)
    scores = torch.stack(scores, -1) # 70 x 100
    print("scores size: ", scores.shape)
    closest_dists, indices = torch.sort(scores, dim=-1, descending=True)
    closest_indices = torch.gather(medium_nn_inds, -1, indices)
    ranks = deepcopy(medium_nn_inds)
    ranks[:, :top_k] = deepcopy(closest_indices)
    ranks = ranks.cpu().data.numpy().T
    # pickle_save('medium_nn_inds.pkl', ranks.T)
    medium = compute_metrics('revisited', ranks, gnd['gnd'], kappas=ks)

    ########################################################################################
    ## Hard
    """
    hard_nn_inds = deepcopy(cache_nn_inds.cpu().data.numpy())
    # Exclude the junk images as in DELG (https://github.com/tensorflow/models/blob/44cad43aadff9dd12b00d4526830f7ea0796c047/research/delf/delf/python/detect_to_retrieve/image_reranking.py#L190)
    for i in range(num_samples):
        junk_ids = gnd['gnd'][i]['junk'] + gnd['gnd'][i]['easy']
        all_ids = hard_nn_inds[i]
        pos = np.in1d(all_ids, junk_ids)
        neg = np.array([not x for x in pos])
        new_ids = np.concatenate([np.arange(len(all_ids))[neg], np.arange(len(all_ids))[pos]])
        new_ids = all_ids[new_ids]
        hard_nn_inds[i] = new_ids
    hard_nn_inds = torch.from_numpy(hard_nn_inds)

    scores = []
    for i in tqdm(range(top_k)):
        nnids = hard_nn_inds[:, i]
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
    scores = torch.stack(scores, -1) # 70 x 100
    closest_dists, indices = torch.sort(scores, dim=-1, descending=True)
    closest_indices = torch.gather(hard_nn_inds, -1, indices)

    # pickle_save('nn_inds_rerank.pkl', closest_indices)
    # pickle_save('nn_dists_rerank.pkl', closest_dists)

    ranks = deepcopy(hard_nn_inds)
    ranks[:, :top_k] = deepcopy(closest_indices)
    ranks = ranks.cpu().data.numpy().T
    # pickle_save('hard_nn_inds.pkl', ranks.T)
    hard = compute_metrics('revisited', ranks, gnd['gnd'], kappas=ks)

    """
    ########################################################################################  
    out = {
        'M_map': float(medium['M_map']), 
        #'H_map': float(hard['H_map']),
        'M_mp':  medium['M_mp'].tolist(),
        #'H_mp': hard['H_mp'].tolist(),
    }
    # json_save('eval_revisited.json', out)
    return out


@torch.no_grad()
def mean_average_precision_revisited_rerank_time(
    model: nn.Module,
    cache_nn_inds: torch.Tensor,
    query_global: torch.Tensor, query_local: torch.Tensor, query_mask: torch.Tensor, query_scales: torch.Tensor, query_positions: torch.Tensor,
    gallery_global: torch.Tensor, gallery_local: torch.Tensor, gallery_mask: torch.Tensor, gallery_scales: torch.Tensor, gallery_positions: torch.Tensor,
    ks: List[int],
    gnd) -> Dict[str, float]:

    device = next(model.parameters()).device
    # query_global    = query_global.to(device)
    # query_local     = query_local.to(device)
    # query_mask      = query_mask.to(device)
    # query_scales    = query_scales.to(device)
    # query_positions = query_positions.to(device)

    num_samples, top_k = cache_nn_inds.size()
    top_k = min(10, top_k)

    ########################################################################################
    ## Medium
    medium_nn_inds = deepcopy(cache_nn_inds.cpu().data.numpy())
    for i in range(num_samples):
        junk_ids = gnd['gnd'][i]['junk']
        all_ids = medium_nn_inds[i]
        pos = np.in1d(all_ids, junk_ids)
        neg = np.array([not x for x in pos])
        new_ids = np.concatenate([np.arange(len(all_ids))[neg], np.arange(len(all_ids))[pos]])
        new_ids = all_ids[new_ids]
        medium_nn_inds[i] = new_ids
    medium_nn_inds = torch.from_numpy(medium_nn_inds)
    
    scores = []
    total_time = 0.0
    for i in range(num_samples):
        nnids = medium_nn_inds[i,:top_k]
        index_global    = gallery_global[nnids]
        index_local     = gallery_local[nnids]
        index_mask      = gallery_mask[nnids]
        index_scales    = gallery_scales[nnids]
        index_positions = gallery_positions[nnids]


        src_global    = query_global[i].unsqueeze(0).repeat(top_k, 1)
        src_local     = query_local[i].unsqueeze(0).repeat(top_k, 1, 1)
        src_mask      = query_mask[i].unsqueeze(0).repeat(top_k, 1)
        src_scales    = query_scales[i].unsqueeze(0).repeat(top_k, 1)
        src_positions = query_positions[i].unsqueeze(0).repeat(top_k, 1, 1)

        # index_global = index_global.to(device)
        # index_local = index_local.to(device)
        # index_mask = index_mask.to(device)
        # index_scales = index_scales.to(device)
        # index_positions = index_positions.to(device)

        start = time.time()
        current_scores = model(
            src_global, src_local, src_mask, src_scales, src_positions,
            index_global,
            index_local,
            index_mask,
            index_scales,
            index_positions)
        end = time.time()
        total_time += end-start
        scores.append(current_scores.cpu().data)
    scores = torch.stack(scores, 0) # 70 x 100
    print('scores', scores.shape)
    print('time', total_time/num_samples)
    closest_dists, indices = torch.sort(scores, dim=-1, descending=True)
    closest_indices = torch.gather(medium_nn_inds, -1, indices)
    ranks = deepcopy(medium_nn_inds)
    ranks[:, :top_k] = deepcopy(closest_indices)
    ranks = ranks.cpu().data.numpy().T
    medium = compute_metrics('revisited', ranks, gnd['gnd'], kappas=ks)

    ########################################################################################  
    out = {
        'M_map': float(medium['M_map']), 
        'M_mp':  medium['M_mp'].tolist(),
    }
    # json_save('eval_revisited.json', out)
    return out