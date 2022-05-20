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
    set_name = 'tuto'
    prefix = None

    save_nn_inds = False
    

@ex.automain
def main(data_dir, set_name, prefix, save_nn_inds):
    
    gnd_name = 'gnd_'+set_name+'.pkl' if prefix is None else prefix+'_gnd_'+set_name+'.pkl'
    gnd_data = pickle_load(osp.join(data_dir, gnd_name))
    sizes = [len(gnd_data['simlist'][i]) for i in range(len(gnd_data['simlist']))]
    
    nn_inds, _ = np.meshgrid(np.arange(100), np.arange(len(gnd_data['qimlist'])))

    for i in range(nn_inds.shape[0]):
        nn_inds[i,:sizes[i]] = gnd_data['gnd'][i]['r_ir_order']
    
    if save_nn_inds:
        if use_aqe:
            output_file = set_name +'_ir_aqe_nn_inds.pkl'
        else:
            output_file = set_name + '_ir_nn_inds.pkl'
        
        output_path = osp.join(data_dir, output_file)
        pickle_save(output_path, nn_inds)
    
    gnd_data = pickle_load(osp.join(data_dir, gnd_name))
    compute_metrics('viquae', nn_inds.T, gnd_data['gnd'], sizes, kappas=[1,5,10])
    