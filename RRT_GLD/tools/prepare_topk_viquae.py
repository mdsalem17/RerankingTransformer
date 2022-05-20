import _init_paths
import os.path as osp
import numpy as np
from tqdm import tqdm
from utils import pickle_save, pickle_load
from utils.data.delf import datum_io
from copy import deepcopy

import sacred
from sacred import SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds
from numpy import linalg as LA
from utils import pickle_load, pickle_save
from utils.revisited import compute_metrics
from utils.data.delf import datum_io


ex = sacred.Experiment('Prepare Top-K (VIQUAE FOR RTT)')
# Filter backspaces and linefeeds
SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def config():
    dataset_name = 'viquae_for_rrt'
    data_dir = osp.join('data', dataset_name)
    feature_name = 'r50_gldv2'
    set_name = 'tuto'
    prefix = None
    
    use_aqe = False
    aqe_params = {'k': 2, 'alpha': 0.3}

    save_nn_inds = True
    

@ex.automain
def main(data_dir, feature_name, set_name, use_aqe, aqe_params, prefix, save_nn_inds):
    
    gnd_name = 'gnd_'+set_name+'.pkl' if prefix is None else prefix+'_gnd_'+set_name+'.pkl'
    
    query_file     = set_name+'_query.txt' if prefix is None else prefix+'_'+set_name+'_query.txt'
    gallery_file   = set_name+'_gallery.txt' if prefix is None else prefix+'_'+set_name+'_gallery.txt'
    selection_file = set_name+'_selection.txt' if prefix is None else prefix+'_'+set_name+'_selection.txt'
    
    with open(osp.join(data_dir, query_file)) as fid:
        query_lines   = fid.read().splitlines()
    #with open(osp.join(data_dir, gallery_file)) as fid:
    #    gallery_lines = fid.read().splitlines()
    with open(osp.join(data_dir, selection_file)) as fid:
        selection_lines = fid.read().splitlines()
        
    query_feats = []
    for i in tqdm(range(len(query_lines))):
        name = osp.splitext(osp.basename(query_lines[i].split(';;')[0]))[0]
        path = osp.join(data_dir, 'delg_' + feature_name, name + '.delg_global')
        query_feats.append(datum_io.ReadFromFile(path))

    query_feats = np.stack(query_feats, axis=0)
    query_feats = query_feats / LA.norm(query_feats, axis=-1)[:, None]

    index_feats = []
    for i in tqdm(range(len(selection_lines))):
        name = osp.splitext(osp.basename(selection_lines[i].split(';;')[0]))[0]
        path = osp.join(data_dir, 'delg_'+feature_name, name+'.delg_global')
        index_feats.append(datum_io.ReadFromFile(path))
        
    selection_index_feats = np.zeros((query_feats.shape[0], 100, query_feats.shape[1]))
    
    gnd_data = pickle_load(osp.join(data_dir, gnd_name))
    selection_index_sizes = [len(gnd_data['simlist'][i]) for i in range(len(gnd_data['simlist']))]
    
    size = 0
    counter = 0
    for i in range(selection_index_feats.shape[0]):
        for j in range(selection_index_feats.shape[1]):
            if j < selection_index_sizes[i]:
                selection_index_feats[i][j] = index_feats[counter]
                counter += 1
    
    for i in range(selection_index_feats.shape[0]):
        selection_index_feats[i] = selection_index_feats[i]/LA.norm(selection_index_feats[i], axis=-1)[:,None]
    
    sims = []
    for i in range(len(selection_index_feats)):
        index_feats = np.stack(selection_index_feats[i], axis=0)
        sims.append(np.matmul(query_feats[i], index_feats.T))

    sims = np.stack(sims, axis=0)
    
    if use_aqe:
        alpha = aqe_params['alpha']
        nn_inds = np.argsort(-sims, -1)
        query_aug = deepcopy(query_feats)
        for i in range(len(query_feats)):
            new_q = [query_feats[i]]
            for j in range(aqe_params['k']):
                nn_id = nn_inds[i, j]
                weight = sims[i, nn_id] ** aqe_params['alpha']
                new_q.append(weight * index_feats[nn_id])
            new_q = np.stack(new_q, 0)
            new_q = np.mean(new_q, axis=0)
            query_aug[i] = new_q/LA.norm(new_q, axis=-1)
        sims = np.matmul(query_aug, index_feats.T)

    nn_inds = np.argsort(-sims, -1)
    nn_dists = deepcopy(sims)
    for i in range(query_feats.shape[0]):
        for j in range(index_feats.shape[0]):
            nn_dists[i, j] = sims[i, nn_inds[i, j]]

    if save_nn_inds:
        if use_aqe:
            output_file = set_name +'_aqe_nn_inds_%s.pkl' % feature_name
        else:
            output_file = set_name + '_nn_inds_%s.pkl' % feature_name
        
        output_file = output_file if prefix is None else prefix+'_'+output_file
        output_path = osp.join(data_dir, output_file)
        pickle_save(output_path, nn_inds)
    
    gnd_data = pickle_load(osp.join(data_dir, gnd_name))
    compute_metrics('viquae', nn_inds.T, gnd_data['gnd'], selection_index_sizes, kappas=[1,5,10])
    