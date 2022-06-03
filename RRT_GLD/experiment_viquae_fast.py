import os, math
import os.path as osp
from copy import deepcopy
from functools import partial
from pprint import pprint

import sacred
import torch
import torch.nn as nn
from sacred import SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds
from torch.backends import cudnn
from torch.optim import SGD, Adam, AdamW, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
# from visdom_logger import VisdomLogger

from models.ingredient import model_ingredient, get_model
from utils import state_dict_to_cpu, num_of_trainable_params
from utils import pickle_load
from utils import BinaryCrossEntropyWithLogits
from utils.data.dataset_ingredient import data_ingredient, get_loaders
from utils.training import train_one_epoch, fast_evaluate_viquae

ex = sacred.Experiment('RRT Training', ingredients=[data_ingredient, model_ingredient])
# Filter backspaces and linefeeds
SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds

@ex.config
def config():
    epochs = 100
    lr = 1e-4
    momentum = 0.
    nesterov = False
    weight_decay = 5e-5
    optim = 'adamw'
    scheduler = 'multistep'
    max_norm = 0.0
    seed = 0

    visdom_port = None
    visdom_freq = 100
    cpu = False  # Force training on CPU
    cudnn_flag = 'benchmark'
    temp_file = 'temp'

    no_bias_decay = False
    loss = 'bce'
    scheduler_tau = [10, 20]
    scheduler_gamma = 0.1

    resume = None
    classifier = False
    transformer = False
    last_layers = False
    scheduling = False


@ex.capture
def get_optimizer_scheduler(parameters, optim, loader_length, epochs, lr, momentum, nesterov, weight_decay, scheduler, scheduler_tau, scheduler_gamma, lr_step=None):
    if optim == 'sgd':
        optimizer = SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True if nesterov and momentum else False)
    elif optim == 'adam':
        optimizer = Adam(parameters, lr=lr, weight_decay=weight_decay) 
    else:
        optimizer = AdamW(parameters, lr=lr, weight_decay=weight_decay)
    
    if epochs == 0:
        scheduler = None
        update_per_iteration = None
    elif scheduler == 'cos':
        # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * loader_length, eta_min=0.000005)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.000001)
        update_per_iteration = False
    elif scheduler == 'warmcos':
        # warm_cosine = lambda i: min((i + 1) / 3, (1 + math.cos(math.pi * i / (epochs * loader_length))) / 2)
        warm_cosine = lambda i: min((i + 1) / 3, (1 + math.cos(math.pi * i / epochs)) / 2)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_cosine)
        update_per_iteration = False
    elif scheduler == 'multistep':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_tau, gamma=scheduler_gamma)
        update_per_iteration = False
    elif scheduler == 'warmstep':
        warm_step = lambda i: min((i + 1) / 100, 1) * 0.1 ** (i // (lr_step * loader_length))
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_step)
        update_per_iteration = True
    else:
        scheduler = lr_scheduler.StepLR(optimizer, epochs * loader_length)
        update_per_iteration = True
 
    return optimizer, (scheduler, update_per_iteration)


@ex.capture
def get_loss(loss):
    if loss == 'bce':
        return BinaryCrossEntropyWithLogits()
    else:
        raise Exception('Unsupported loss {}'.format(loss))


@ex.automain
def main(epochs, cpu, cudnn_flag, visdom_port, visdom_freq, temp_file, seed, no_bias_decay, max_norm, classifier, transformer, last_layers, resume, scheduling):
    device = torch.device('cuda:0' if torch.cuda.is_available() and not cpu else 'cpu')
    temp_dir = osp.join('outputs', temp_file)
    logs_dir = osp.join('logs', temp_file)
    writer = SummaryWriter(logs_dir)

    # callback = VisdomLogger(port=visdom_port) if visdom_port else None
    if cudnn_flag == 'deterministic':
        setattr(cudnn, cudnn_flag, True)

    torch.manual_seed(seed)
    loaders, recall_ks = get_loaders()

    #torch.manual_seed(seed+1)
    model = get_model()    
    if classifier:
        #freeze all layers of the model
        for param in model.parameters():
            param.requires_grad = False

        #unfreeze the classfication layers
        for param in model.classifier.parameters():
            param.requires_grad = True
        
    if transformer:
        #freeze all layers of the model
        for param in model.parameters():
            param.requires_grad = True
        
        for param in model.classifier.parameters():
            param.requires_grad = False
        for param in model.seg_encoder.parameters():
            param.requires_grad = False
        for param in model.scale_encoder.parameters():
            param.requires_grad = False
        for param in model.remap.parameters():
            param.requires_grad = False
    
    if last_layers:
        #freeze all layers of the model
        for param in model.parameters():
            param.requires_grad = False
        
        for param in model.classifier.parameters():
            param.requires_grad = True
        for param in model.seg_encoder.parameters():
            param.requires_grad = True
        for param in model.scale_encoder.parameters():
            param.requires_grad = True
        for param in model.remap.parameters():
            param.requires_grad = True
    
    if resume is not None:
        checkpoint = torch.load(resume, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state'], strict=True)
    print('# of trainable parameters: ', num_of_trainable_params(model))
    class_loss = get_loss()
    
    nn_inds_path = loaders.eval_set_name + '_nn_inds_%s.pkl'%loaders.query.dataset.desc_name
    nn_inds_path = nn_inds_path if loaders.prefixed is None else loaders.prefixed+'_'+nn_inds_path
    nn_inds_path = osp.join(loaders.query.dataset.data_dir, nn_inds_path)
    print('nn_inds_path:', nn_inds_path)
    cache_nn_inds = torch.from_numpy(pickle_load(nn_inds_path)).long()

    #torch.manual_seed(seed+2)
    model.to(device)
    model = nn.DataParallel(model)
    parameters = []
    if no_bias_decay:
        parameters.append({'params': [par for par in model.parameters() if par.dim() != 1]})
        parameters.append({'params': [par for par in model.parameters() if par.dim() == 1], 'weight_decay': 0})
    else:
        parameters.append({'params': model.parameters()})
    optimizer, scheduler = get_optimizer_scheduler(parameters=parameters, loader_length=len(loaders.train))
    if resume is not None and checkpoint.get('optim', None) is not None:
        optimizer.load_state_dict(checkpoint['optim'])
        del checkpoint

    #torch.manual_seed(seed+3)
    query_feats, gallery_feats = [], []
    #eval_function = partial(evaluate_viquae, model=model, 
    #    cache_nn_inds=cache_nn_inds,
    #    recall=recall_ks, query_loader=loaders.query, gallery_loader=loaders.gallery)

    # setup best validation logger
    result, query_feats, gallery_feats = fast_evaluate_viquae(model=model,
                                                              cache_nn_inds=cache_nn_inds,
                                                              recall=recall_ks,
                                                              query_loader=loaders.query, 
                                                              gallery_loader=loaders.gallery,
                                                              query_feats=query_feats, 
                                                              gallery_feats=gallery_feats)
    # if callback is not None:
    #     callback.scalars(['l2', 'cosine'], 0, [metrics.recall['l2'][1], metrics.recall['cosine'][1]],
    #                      title='Val Recall@1')
    pprint(result)
    best_val = (0, result, deepcopy(model.state_dict()))

    # saving
    save_name = osp.join(temp_dir, '{}_{}.pt'.format(ex.current_run.config['model']['name'],
                                                         ex.current_run.config['dataset']['name']))
    os.makedirs(temp_dir, exist_ok=True)
    #torch.manual_seed(seed+4)
    for epoch in range(epochs):
        if cudnn_flag == 'benchmark':
            setattr(cudnn, cudnn_flag, True)

        torch.cuda.empty_cache()
        train_one_epoch(model=model, loader=loaders.train, class_loss=class_loss, optimizer=optimizer, scheduler=scheduler, max_norm=max_norm, epoch=epoch, freq=visdom_freq, writer=writer, ex=None, scheduling=scheduling)

        # validation
        if cudnn_flag == 'benchmark':
            setattr(cudnn, cudnn_flag, False)
        torch.cuda.empty_cache()
        result, query_feats, gallery_feats = fast_evaluate_viquae(model=model,
                                                              cache_nn_inds=cache_nn_inds,
                                                              recall=recall_ks,
                                                              query_loader=loaders.query, 
                                                              gallery_loader=loaders.gallery,
                                                              query_feats=query_feats, 
                                                              gallery_feats=gallery_feats)
        print('Validation [{:03d}]'.format(epoch)), pprint(result)
        
        ex.log_scalar('val.map', result['map'], step=epoch + 1)
        writer.add_scalar('eval/map', result['map'],  epoch)
        writer.add_scalar('eval/mrr', result['mrr'],    epoch)
        writer.add_scalar('eval/precision@1', result['precision@1'], epoch)
        writer.add_scalar('eval/precision@5', result['precision@5'], epoch)
        
        save_name_epoch = osp.join(temp_dir, '{}_{}_{}.pt'.format(ex.current_run.config['model']['name'],
                                                                  ex.current_run.config['dataset']['name'],
                                                                  epoch))
        
        
        epoch_val = (epoch + 1, result, deepcopy(model.state_dict()))
        torch.save({'state': state_dict_to_cpu(epoch_val[2]), 'optim': optimizer.state_dict()}, save_name_epoch)
        
        if result['map'] >= best_val[1]['map']:
            print('New best model in epoch %d.'%epoch)
            best_val = (epoch + 1, result, deepcopy(model.state_dict()))
            torch.save({'state': state_dict_to_cpu(best_val[2]), 'optim': optimizer.state_dict()}, save_name)

    # logging
    ex.info['metrics'] = best_val[1]
    ex.add_artifact(save_name)
    writer.flush()
    writer.close()
    

    return best_val[1]
