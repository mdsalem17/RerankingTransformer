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


ex = sacred.Experiment('Prepare Top-K (VIQUAE)')
# Filter backspaces and linefeeds
SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def config():
    dataset_name = 'viquae_images'
    data_dir = osp.join('data', dataset_name)
    feature_name = 'r50_gldv1'

    use_aqe = False
    aqe_params = {'k': 2, 'alpha': 0.3}

    save_nn_inds = True
    

@ex.automain
def main(data_dir, feature_name, use_aqe, aqe_params, save_nn_inds):
    with open(osp.join(data_dir, 'test_query.txt')) as fid:
        query_lines   = fid.read().splitlines()
    with open(osp.join(data_dir, 'test_gallery.txt')) as fid:
        gallery_lines = fid.read().splitlines()

    query_feats = []
    for i in tqdm(range(len(query_lines))):
        name = osp.splitext(osp.basename(query_lines[i].split(';;')[0]))[0]
        path = osp.join(data_dir, 'delg_' + feature_name, name + '.delg_global')
        query_feats.append(datum_io.ReadFromFile(path))

    query_feats = np.stack(query_feats, axis=0)
    query_feats = query_feats / LA.norm(query_feats, axis=-1)[:, None]

    index_feats = []
    for i in tqdm(range(len(gallery_lines))):
        name = osp.splitext(osp.basename(gallery_lines[i].split(';;')[0]))[0]
        path = osp.join(data_dir, 'delg_' + feature_name, name + '.delg_global')
        index_feats.append(datum_io.ReadFromFile(path))

    index_feats = np.stack(index_feats, axis=0)
    index_feats = index_feats / LA.norm(index_feats, axis=-1)[:, None]

    sims = np.matmul(query_feats, index_feats.T)

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

    # x_step = 2000
    # y_step = 2000
    # n = len(desc)
    # nn_inds = []
    # for i in tqdm(range(0, n, x_step)):
    #     tmp_pkl_name = osp.join(data_dir, feature_name+'_nn_%09d.pkl'%i)
    #     if osp.exists(tmp_pkl_name):
    #         inds = pickle_load(tmp_pkl_name)
    #         nn_inds.append(inds)
    #         continue
    #     xs = desc[i:min(i+x_step, n)]
    #     sims = []
    #     for j in range(0, n, y_step):
    #         ys = desc[j:min(j+y_step, n)]
    #         ds = np.matmul(xs, ys.T)
    #         sims.append(ds)
    #     sims = np.concatenate(sims, axis=-1)
    #     inds = np.argsort(-sims, axis=-1)[:, 1:101]
    #     pickle_save(tmp_pkl_name, inds)
    #     nn_inds.append(inds)

    nn_inds = np.argsort(-sims, -1)
    nn_dists = deepcopy(sims)
    for i in range(query_feats.shape[0]):
        for j in range(index_feats.shape[0]):
            nn_dists[i, j] = sims[i, nn_inds[i, j]]

    if save_nn_inds:
        output_path = osp.join(data_dir, 'nn_inds_%s.pkl' % feature_name)
        pickle_save(output_path, nn_inds)

    # nn_inds = np.concatenate(nn_inds, 0)
    # print(nn_inds.shape)
    # output_path = osp.join(data_dir, 'nn_inds_%s.pkl'%feature_name)
    # pickle_save(output_path, nn_inds)
