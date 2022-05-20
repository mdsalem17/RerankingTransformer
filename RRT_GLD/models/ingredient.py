from .matcher import MatchERT, PosMatchERT
from sacred import Ingredient
model_ingredient = Ingredient('model')


@model_ingredient.config
def config():
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
    use_pos = False


@model_ingredient.named_config
def RRT():
    name = 'rrt'
    seq_len = 1004
    dim_K = 256
    dim_feedforward = 1024
    nhead = 4
    num_encoder_layers = 6
    dropout = 0.0 
    activation = "relu"
    normalize_before = False



@model_ingredient.named_config
def dRRT():
    name = 'rrt'
    seq_len = 1004
    dim_K = 256
    dim_feedforward = 1024
    nhead = 4
    num_encoder_layers = 6
    dropout = 0.2
    activation = "relu"
    normalize_before = False



@model_ingredient.named_config
def posRRT():
    name = 'posrrt'
    seq_len = 1004
    dim_K = 256
    dim_feedforward = 1024
    nhead = 4
    num_encoder_layers = 6
    dropout = 0.0 
    activation = "relu"
    normalize_before = False
    use_pos = True


@model_ingredient.named_config
def vRRT():
    name = 'vrrt'
    seq_len = 1004
    dim_K = 256
    dim_feedforward = 1024
    nhead = 8
    num_encoder_layers = 8
    dropout = 0.4 
    activation = "relu"
    normalize_before = False


@model_ingredient.named_config
def vPosRRT():
    name = 'vposrrt'
    seq_len = 1004
    dim_K = 256
    dim_feedforward = 1024
    nhead = 8
    num_encoder_layers = 8
    dropout = 0.4 
    activation = "relu"
    use_pos = True


@model_ingredient.capture
def get_model(num_global_features, num_local_features, seq_len, dim_K, dim_feedforward, nhead, num_encoder_layers, dropout, activation, normalize_before, use_pos):
    
    if use_pos:
        return PosMatchERT(d_global=num_global_features, d_model=num_local_features, seq_len=seq_len, d_K=dim_K, nhead=nhead, num_encoder_layers=num_encoder_layers, 
            dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, normalize_before=normalize_before)
    
    else:
        return MatchERT(d_global=num_global_features, d_model=num_local_features, seq_len=seq_len, d_K=dim_K, nhead=nhead, num_encoder_layers=num_encoder_layers, 
            dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, normalize_before=normalize_before)