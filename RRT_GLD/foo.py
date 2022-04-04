 1/1: from datasets import load_from_disk
 1/2: dataset =load_from_disk("meerqat_dataset_v1.0/")
 1/3: dataset
 1/4: item =dataset["test"][0]
 1/5: item["BM25_indices"]
 1/6: dataset
 2/1:
from datasets import load_from_disk, set_caching_enabled
from meerqat.ir.metrics import find_relevant

set_caching_enabled(False)
 3/1: from datasets import load_from_disk, set_caching_enabled
 3/2: from meerqat.ir.metrics import find_relevant
 3/3: set_caching_enabled(False)
 3/4: kb = load_from_disk('data/viquae_passages/')
 3/5: dataset = load_from_disk('data/viquae_dataset/')
 3/6: dataset
 3/7: item = dataset['test'][0]
 3/8: type(item)
 3/9: item['BM25_indices']
3/10: dataset
3/11:
def keep_relevant_search_wrt_original_in_priority(item, kb):
    # this contains the latest result of the fusion
    # to reproduce the results of the paper, use DPR+Image as IR
    indices = item['search_indices']
    relevant_indices, _ = find_relevant(indices, item['output']['original_answer'], [], kb)
    if relevant_indices:
        item['search_provenance_indices'] = relevant_indices
    else:
        item['search_provenance_indices'] = item['original_answer_provenance_indices']
    item['search_irrelevant_indices'] = list(set(indices) - set(relevant_indices))
    return item
3/12: dataset = dataset.map(keep_relevant_search_wrt_original_in_priority, fn_kwargs=dict(kb=kb))
3/13: item = dataset[0]
 4/1:
from datasets import load_from_disk, set_caching_enabled
from meerqat.ir.metrics import find_relevant
 4/2: ls
 4/3: dataset = load_from_disk('data/viquae_dataset/test')
 4/4: item = dataset[0]
 4/5: import ranx
 4/6: run = ranx.Run.from_file('experiments/ir/viquae/dpr+arcface+clip+imagenet/test/fusion.trec')
 4/7: run = ranx.Run.from_file('experiments/ir/viquae/dpr+arcface+clip+imagenet/test_set/fusion.trec')
 5/1: import ranx
 5/2: from datasets import load_from_disk, set_caching_enabled
 5/3: set_caching_enabled(False)
 5/4: dataset = load_from_disk('data/viquae_dataset/test')
 5/5: run = ranx.Run.from_file('experiments/ir/viquae/dpr+arcface+clip+imagenet/test_set/fusion.trec')
 5/6: ?ranx.Run.from_file
 5/7: run = ranx.Run.from_file('experiments/ir/viquae/dpr+arcface+clip+imagenet/test_set/fusion.trec', kind="trec")
 5/8: run
 5/9: run["ffc27e81f5f86dd432e8bb2cc820dffe"].keys()
5/10: list(map(int,run["ffc27e81f5f86dd432e8bb2cc820dffe"].keys()))
5/11: ranx_indices = list(map(int,run["ffc27e81f5f86dd432e8bb2cc820dffe"].keys()))
5/12: dataset = dataset.map(lambda item: {'search_indices': list(map(int,run[item['id']].keys()))})
5/13: dataset.save_disk('data/viquae_dataset/test')
5/14: dataset.save_to_disk('data/viquae_dataset/test')
5/15: ?ranx.Run.from_file
5/16: dataset = load_from_disk('data/viquae_dataset/train')
5/17: run = ranx.Run.from_file('experiments/ir/viquae/dpr+arcface+clip+imagenet/train_set/fusion.trec')
5/18: run = ranx.Run.from_file('experiments/ir/viquae/dpr+arcface+clip+imagenet/train_set/fusion.trec', kind="trec")
5/19: item
5/20: dataset = dataset.map(lambda item: {'search_indices': list(map(int,run[item['id']].keys()))})
5/21: dataset.save_to_disk('data/viquae_dataset/train')
5/22: dataset = load_from_disk('data/viquae_dataset/validation')
5/23: run = ranx.Run.from_file('experiments/ir/viquae/dpr+arcface+clip+imagenet/dev_set/fusion.trec', kind="trec")
5/24: dataset = dataset.map(lambda item: {'search_indices': list(map(int,run[item['id']].keys()))})
5/25: dataset.save_to_disk('data/viquae_dataset/validation')
5/26: from datasets import load_from_disk, set_caching_enabled
5/27: from meerqat.ir.metrics import find_relevant
5/28: set_caching_enabled(False)
5/29: kb = load_from_disk('data/viquae_passages/')
5/30: dataset = load_from_disk('data/viquae_dataset/')
5/31:
def keep_relevant_search_wrt_original_in_priority(item, kb):
    # this contains the latest result of the fusion
    # to reproduce the results of the paper, use DPR+Image as IR
    indices = item['search_indices']
    relevant_indices, _ = find_relevant(indices, item['output']['original_answer'], [], kb)
    if relevant_indices:
        item['search_provenance_indices'] = relevant_indices
    else:
        item['search_provenance_indices'] = item['original_answer_provenance_indices']
    item['search_irrelevant_indices'] = list(set(indices) - set(relevant_indices))
    return item
5/32: dataset = dataset.map(keep_relevant_search_wrt_original_in_priority, fn_kwargs=dict(kb=kb))
5/33: dataset.save_to_disk('data/viquae_dataset/')
 6/1: %tensorboard --logdir=runs
 6/2: %load_ext tensorboard
 6/3: %load_ext tensorboard.notebook
 6/4: %load_ext tensorboard
 6/5: !pwd
 6/6: %tensorboard --logdir=runs
 7/1: %load_ext tensorboard
 7/2: %tensorboard --logdir=runs
 8/1: %load_ext tensorboard
 8/2: pwdw
 8/3: pwd
 8/4: %tensorboard --logdir=experiments/rc/viquae/train
 8/5: %tensorboard --logdir=experiments/rc/viquae/train --port=8888
 9/1: from datasets import load_from_disk, set_caching_enabled
 9/2: import ranx
 9/3: dataset = load_from_disk('data/viquae_dataset/test')
 9/4: run = ranx.Run.from_file('experiments/ir/viquae/dpr+arcface+clip+imagenet/test_set/fusion.trec', kind="trec")
 9/5: dataset = dataset.map(lambda item: {'search_indices': list(map(int,run[item['id']].keys()))})
 9/6: dataset = dataset.map(keep_relevant_search_wrt_original_in_priority, fn_kwargs=dict(kb=kb))
 9/7: dataset = dataset.map(lambda item: {'search_indices': list(map(int,run[item['id']].keys()))})
 9/8: dataset = load_from_disk('data/viquae_dataset/test')
 9/9: run = ranx.Run.from_file('experiments/ir/viquae/dpr+arcface+clip+imagenet/test_set/fusion.trec', kind="trec")
9/10: dataset = dataset.map(lambda item: {'search_indices': list(map(int,run[item['id']].keys()))})
9/11: dataset.save_to_disk('data/viquae_dataset/test')
9/12: dataset = load_from_disk('data/viquae_dataset/validation')
9/13: run = ranx.Run.from_file('experiments/ir/viquae/dpr+arcface+clip+imagenet/dev_set/fusion.trec', kind="trec")
9/14: dataset = dataset.map(lambda item: {'search_indices': list(map(int,run[item['id']].keys()))})
9/15: dataset.save_to_disk('data/viquae_dataset/validation')
9/16: dataset = load_from_disk('data/viquae_dataset/train')
9/17: run = ranx.Run.from_file('experiments/ir/viquae/dpr+arcface+clip+imagenet/train_set/fusion.trec', kind="trec")
9/18: dataset = dataset.map(lambda item: {'search_indices': list(map(int,run[item['id']].keys()))})
9/19: dataset.save_to_disk('data/viquae_dataset/validation')
9/20: dataset.save_to_disk('data/viquae_dataset/train')
9/21: dataset = load_from_disk('data/viquae_dataset/validation')
9/22: run = ranx.Run.from_file('experiments/ir/viquae/dpr+arcface+clip+imagenet/dev_set/fusion.trec', kind="trec")
9/23: dataset = dataset.map(lambda item: {'search_indices': list(map(int,run[item['id']].keys()))})
9/24: dataset.save_to_disk('data/viquae_dataset/validation')
9/25: from meerqat.ir.metrics import find_relevant
9/26: set_caching_enabled(False)
9/27: kb = load_from_disk('data/viquae_passages/')
9/28: dataset = load_from_disk('data/viquae_dataset/')
9/29:
def keep_relevant_search_wrt_original_in_priority(item, kb):
    # this contains the latest result of the fusion
    # to reproduce the results of the paper, use DPR+Image as IR
    indices = item['search_indices']
    relevant_indices, _ = find_relevant(indices, item['output']['original_answer'], [], kb)
    if relevant_indices:
        item['search_provenance_indices'] = relevant_indices
    else:
        item['search_provenance_indices'] = item['original_answer_provenance_indices']
    item['search_irrelevant_indices'] = list(set(indices) - set(relevant_indices))
    return item
9/30:
dataset = dataset.map(keep_relevant_search_wrt_original_in_priority, fn_kwargs=dict(kb=kb))
dataset.save_to_disk('data/viquae_dataset/')
9/31: dataset.save_to_disk('data/viquae_dataset/')
10/1: import ranx
10/2: from datasets import load_from_disk, set_caching_enabled
10/3: dataset = load_from_disk('data/viquae_dataset/test')
10/4: run = ranx.Run.from_file('experiments/ir/viquae/dpr+arcface+clip+imagenet/test_set/fusion.trec', kind="trec")
10/5: dataset = dataset.map(lambda item: {'search_indices': list(map(int,run[item['id']].keys()))})
10/6: dataset.save_to_disk('data/viquae_dataset/test')
10/7: dataset = load_from_disk('data/viquae_dataset/validation')
10/8: run = ranx.Run.from_file('experiments/ir/viquae/dpr+arcface+clip+imagenet/dev_set/fusion.trec', kind="trec")
10/9: dataset = dataset.map(lambda item: {'search_indices': list(map(int,run[item['id']].keys()))})
10/10: dataset.save_to_disk('data/viquae_dataset/validation')
10/11: dataset = load_from_disk('data/viquae_dataset/train')
10/12: run = ranx.Run.from_file('experiments/ir/viquae/dpr+arcface+clip+imagenet/train_set/fusion.trec', kind="trec")
10/13: dataset = dataset.map(lambda item: {'search_indices': list(map(int,run[item['id']].keys()))})
10/14: dataset.save_to_disk('data/viquae_dataset/train')
10/15: from meerqat.ir.metrics import find_relevant
10/16: set_caching_enabled(False)
10/17: kb = load_from_disk('data/viquae_passages/')
10/18: dataset = load_from_disk('data/viquae_dataset/')
10/19:
def keep_relevant_search_wrt_original_in_priority(item, kb):
    # this contains the latest result of the fusion
    # to reproduce the results of the paper, use DPR+Image as IR
    indices = item['search_indices']
    relevant_indices, _ = find_relevant(indices, item['output']['original_answer'], [], kb)
    if relevant_indices:
        item['search_provenance_indices'] = relevant_indices
    else:
        item['search_provenance_indices'] = item['original_answer_provenance_indices']
    item['search_irrelevant_indices'] = list(set(indices) - set(relevant_indices))
    return item
10/20: dataset = dataset.map(keep_relevant_search_wrt_original_in_priority, fn_kwargs=dict(kb=kb))
10/21: dataset.save_to_disk('data/viquae_dataset/')
11/1: from datasets import load_from_disk, set_caching_enabled
11/2: dataset = load_from_disk("data/viquae_dataset/test")
12/1: from datasets import load_from_disk, set_caching_enabled
12/2: dataset = load_from_disk("data/viquae_dataset/test")
12/3: from transformers import Automodel
13/1: from datasets import load_from_disk, set_caching_enabled
13/2: set_caching_enabled(False)
13/3: dataset = load_from_disk('data/viquae_dataset/test')
13/4: from meerqat.train.trainee import MultiPassageBERT
13/5: config = load_pretrained_in_kwargs(json.load("experiments/rc/viquae_6/test/config.json"))
13/6: from meerqat.data.loading import load_pretrained_in_kwargs
13/7: config = load_pretrained_in_kwargs(json.load("experiments/rc/viquae_6/test/config.json"))
13/8: import json
13/9: config = load_pretrained_in_kwargs(json.load("experiments/rc/viquae_6/test/config.json"))
14/1: import json
14/2: from meerqat.data.loading import load_pretrained_in_kwargs
14/3: from meerqat.train.trainee import MultiPassageBERT
14/4: from datasets import load_from_disk, set_caching_enabled
14/5: from pathlib import Path
14/6: config_path = Path("experiments/rc/viquae_6/test/config.json")
14/7:
with open(config_path, "r") as file:
    config = load_pretrained_in_kwargs(json.load(file))
14/8: from meerqat.train.trainer import instantiate_trainer
14/9: trainer, training_args = instantiate_trainer(**config)
14/10: verbosity = config.pop("verbosity", None)
14/11: trainer, training_args = instantiate_trainer(**config)
14/12: checkpoint = config.pop("checkpoint", {})
14/13: trainer, training_args = instantiate_trainer(**config)
14/14: trainer.args.device
14/15: device = trainer.args.device
14/16: resume_from_checkpoints = get_checkpoint(**checkpoint)
14/17: from meerqat.train.trainer import get_checkpoint
14/18: resume_from_checkpoints = get_checkpoint(**checkpoint)
14/19: from transformers.file_utils import WEIGHTS_NAME
14/20:
for resume_from_checkpoint in tqdm(resume_from_checkpoints, desc="Prediction"):
    state_dict_path = resume_from_checkpoint / WEIGHTS_NAME
    if not state_dict_path.exists():
        continue
14/21: from tqdm import tqdm
14/22:
for resume_from_checkpoint in tqdm(resume_from_checkpoints, desc="Prediction"):
    state_dict_path = resume_from_checkpoint / WEIGHTS_NAME
    if not state_dict_path.exists():
        continue
14/23:
for resume_from_checkpoint in tqdm(resume_from_checkpoints, desc="Prediction"):
    state_dict_path = resume_from_checkpoint / WEIGHTS_NAME
    if not state_dict_path.exists():
        continue
    state_dict = torch.load(state_dict_path, map_locatiion=device)
    trainer._load_state_dict_in_model(state_dict)
    prediction_ouput = trainer.predict(trainer.eval_dataset)
14/24: import torch
14/25:
for resume_from_checkpoint in tqdm(resume_from_checkpoints, desc="Prediction"):
    state_dict_path = resume_from_checkpoint / WEIGHTS_NAME
    if not state_dict_path.exists():
        continue
    state_dict = torch.load(state_dict_path, map_locatiion=device)
    trainer._load_state_dict_in_model(state_dict)
    prediction_ouput = trainer.predict(trainer.eval_dataset)
14/26:
for resume_from_checkpoint in tqdm(resume_from_checkpoints, desc="Prediction"):
    state_dict_path = resume_from_checkpoint / WEIGHTS_NAME
    if not state_dict_path.exists():
        continue
    state_dict = torch.load(state_dict_path, map_locatiion=device)
    trainer._load_state_dict_in_model(state_dict)
    prediction_ouput = trainer.predict(trainer.eval_dataset)
14/27:
for resume_from_checkpoint in tqdm(resume_from_checkpoints, desc="Prediction"):
    state_dict_path = resume_from_checkpoint / WEIGHTS_NAME
    if not state_dict_path.exists():
        continue
    state_dict = torch.load(state_dict_path, map_location=device)
    trainer._load_state_dict_in_model(state_dict)
    prediction_ouput = trainer.predict(trainer.eval_dataset)
14/28: prediction_output
14/29: prediction_ouput
14/30: prediction_ouput['0']
14/31: prediction_ouput[0]
14/32: prediction_ouput[0][0]
14/33: prediction_ouput[1][0]
14/34: prediction_ouput[1]
14/35: prediction_ouput[2]
14/36: prediction_ouput.size
14/37: prediction_ouput.size()
14/38: len(prediction_ouput)
14/39: prediction_ouput[0][1]
14/40: eval_dataset = load_from_disk("data/viquae_dataset/test")
14/41: eval_dataset
14/42: eval_dataset[0]
14/43: item = eval_dataset[0]
14/44: type(item)
14/45: item['original_answer_provenance_indices']
14/46: item['original_answer']
14/47: eval_dataset
14/48: item['original_question']
14/49: item['output']
14/50: prediction_ouput[0][1]
14/51: item['original_question']
15/1: import ranx
15/2: from datasets import load_from_disk, set_caching_enabled
15/3: dataset = load_from_disk('data/viquae_dataset/test')
15/4: run = ranx.Run.from_file('experiments/ir/viquae/dpr+arcface+clip+imagenet/test_set/fusion.trec', kind="trec")
15/5: dataset = dataset.map(lambda item: {'search_indices': list(map(int,run[item['id']].keys()))})
17/1: import ranx
17/2: from datasets import load_from_disk, set_caching_enabled
17/3: dataset = dataset.map(lambda item: {'search_indices': list(map(int,run[item['id']].keys()))})
17/4: dataset = dataset.map(lambda item: {'search_indices': list(map(int,run[item['id']].keys()))})
17/5: dataset = load_from_disk('data/viquae_dataset/test')
18/1: import ranx
18/2: dataset = load_from_disk('data/viquae_dataset/test')
18/3: from datasets import load_from_disk, set_caching_enabled
18/4: dataset = load_from_disk('data/viquae_dataset/test')
18/5: dataset = dataset.map(lambda item: {'search_indices': list(map(int,run[item['id']].keys()))})
18/6: run = ranx.Run.from_file('experiments/ir/viquae/dpr+arcface+clip+imagenet/test_set/fusion.trec', kind="trec")
18/7: dataset = dataset.map(lambda item: {'search_indices': list(map(int,run[item['id']].keys()))})
18/8: dataset.save_to_disk('data/viquae_dataset/test')
18/9: dataset = load_from_disk('data/viquae_dataset/validation')
18/10: run = ranx.Run.from_file('experiments/ir/viquae/dpr+arcface+clip+imagenet/dev_set/fusion.trec', kind="trec")
18/11: dataset = dataset.map(lambda item: {'search_indices': list(map(int,run[item['id']].keys()))})
18/12: dataset.save_to_disk('data/viquae_dataset/validation')
18/13: dataset = load_from_disk('data/viquae_dataset/train')
18/14: run = ranx.Run.from_file('experiments/ir/viquae/dpr+arcface+clip+imagenet/train_set/fusion.trec', kind="trec")
18/15: dataset = dataset.map(lambda item: {'search_indices': list(map(int,run[item['id']].keys()))})
18/16: dataset.save_to_disk('data/viquae_dataset/train')
18/17: from meerqat.ir.metrics import find_relevant
18/18: set_caching_enabled(False)
18/19:
def keep_relevant_search_wrt_original_in_priority(item, kb):
    # this contains the latest result of the fusion
    # to reproduce the results of the paper, use DPR+Image as IR
    indices = item['search_indices']
    relevant_indices, _ = find_relevant(indices, item['output']['original_answer'], [], kb)
    if relevant_indices:
        item['search_provenance_indices'] = relevant_indices
    else:
        item['search_provenance_indices'] = item['original_answer_provenance_indices']
    item['search_irrelevant_indices'] = list(set(indices) - set(relevant_indices))
    return item
18/20: dataset = dataset.map(keep_relevant_search_wrt_original_in_priority, fn_kwargs=dict(kb=kb))
18/21: kb = load_from_disk('data/viquae_passages/')
18/22: dataset = load_from_disk('data/viquae_dataset/')
18/23: dataset = dataset.map(keep_relevant_search_wrt_original_in_priority, fn_kwargs=dict(kb=kb))
18/24: dataset.save_to_disk('data/viquae_dataset/')
19/1: from datasets import load_from_disk, set_caching_enabled
19/2: dataset = load_from_disk("data/viquae_dataset/test")
19/3: set_caching_enabled(False)
19/4: from meerqat.train.trainee import MultiPassageBERT
19/5: config = load_pretrained_in_kwargs(json.load("experiments/rc/viquae_6/test/config.json"))
21/1: from datasets import load_from_disk, set_caching_enabled
21/2: from meerqat.train.trainee import MultiPassageBERT
21/3: from meerqat.data.loading import load_pretrained_in_kwargs
21/4: set_caching_enabled(False)
21/5: dataset = load_from_disk("data/viquae_dataset/test")
21/6: config = load_pretrained_in_kwargs(json.load("experiments/rc/viquae_6/test/config.json"))
21/7: import json
21/8: config = load_pretrained_in_kwargs(json.load("experiments/rc/viquae_6/test/config.json"))
22/1: import json
22/2: import torch
22/3: from pathlib import Path
22/4: from datasets import load_from_disk, set_caching_enabled
22/5:
from meerqat.train.trainee import MultiPassageBERT
from meerqat.data.loading import load_pretrained_in_kwargs
from meerqat.train.trainer import instantiate_trainer
from meerqat.train.trainer import get_checkpoint
from transformers.file_utils import WEIGHTS_NAME
22/6: set_caching_enabled(False)
22/7: config_path = Path("experiments/rc/viquae_6/test/config.json")
22/8:
with open(config_path, "r") as file: 
    config = load_pretrained_in_kwargs(json.load(file))
22/9: verbosity = config.pop("verbosity", None)
22/10: checkpoint = config.pop("checkpoint", {})
22/11: trainer, training_args = instantiate_trainer(**config)
22/12: device = trainer.args.device
22/13: resume_from_checkpoints = get_checkpoint(**checkpoint)
22/14:
for resume_from_checkpoint in tqdm(resume_from_checkpoints, desc="Prediction"): 
    state_dict_path = resume_from_checkpoint / WEIGHTS_NAME 
    if not state_dict_path.exists(): 
        continue 
       
    state_dict = torch.load(state_dict_path, map_location=device)
    trainer._load_state_dict_in_model(state_dict)
    prediction_output = trainer.predict(trainer.eval_dataset)
22/15: from tqdm import tqdm
22/16:
for resume_from_checkpoint in tqdm(resume_from_checkpoints, desc="Prediction"): 
    state_dict_path = resume_from_checkpoint / WEIGHTS_NAME 
    if not state_dict_path.exists(): 
        continue 
       
    state_dict = torch.load(state_dict_path, map_location=device)
    trainer._load_state_dict_in_model(state_dict)
    prediction_output = trainer.predict(trainer.eval_dataset)
22/17: prediction_output.metrics
22/18: type(prediction_output)
22/19: prediction_output.predictions
22/20: prediction_output.predictions[:10]
22/21: trainer.eval_dataset
22/22: trainer.eval_dataset['original_question']
22/23: trainer.eval_dataset['original_question'][:10]
22/24: prediction_output.predictions[:10]
22/25: trainer.eval_dataset
22/26: trainer.eval_dataset['output']
22/27: trainer.eval_dataset
22/28: trainer.eval_dataset['output']['original_answer']
22/29: trainer.eval_dataset['output'][:1]['original_answer']
22/30: trainer.eval_dataset['output'][:1]
22/31: trainer.eval_dataset['output'][0]
22/32: type(trainer.eval_dataset['output'][0])
22/33: trainer.eval_dataset['output'][0]['original_answer']
22/34: trainer.eval_dataset['output'][:10]['original_answer']
22/35: trainer.eval_dataset['output'][:10]
22/36: predicted_answers = prediction_output.predictions[:10]
22/37: predicted_answers
22/38: predicted_answers = [item['prediction_text'] for item in prediction_output.predictions[:10]]
22/39: predicted_answers
22/40: trainer.eval_dataset['output'][:10]
22/41: trainer.eval_dataset['original_question'][:10]
22/42: original_question = trainer.eval_dataset['original_question'][:10]
22/43: original_questions = trainer.eval_dataset['original_question'][:10]
22/44: trainer.eval_dataset['output'][:10]
22/45: original_answers = [item['original_answer'] for item in trainer.eval_dataset['output'][:10]]
22/46: predicted_answers
22/47: original_questions
22/48: original_answers
22/49: trainer.eval_dataset
22/50: trainer.eval_dataset[0]
22/51: item = trainer.eval_dataset[0]
22/52: trainer.predict(item)
22/53: item = trainer.eval_dataset[:10]
22/54: trainer.predict(item)
22/55: checkpoint
22/56: checkpoint['resume_from_checkpoint'] = null
22/57:
with open(config_path, "r") as file: 
    config = load_pretrained_in_kwargs(json.load(file))
22/58: checkpoint = config.pop("checkpoint", {})
22/59: checkpoint
22/60: trainer.train(**checkpoint)
22/61: trainer.eval_dataset
22/62: trainer.eval_dataset['document_search_scores']
22/63: type(trainer.eval_dataset['document_search_scores'])
22/64: len(trainer.eval_dataset['document_search_scores'])
22/65: len(trainer.eval_dataset['document_search_scores'][0])
22/66: trainer.eval_dataset
22/67: len(trainer.eval_dataset['search_scores'])
22/68: trainer.eval_dataset['document_search_scores']
22/69: trainer.eval_dataset['search_scores']
22/70: trainer
22/71: config_path = Path("experiments/rc/viquae_6/train/config.json")
22/72:
with open(config_path, "r") as file: 
    config = load_pretrained_in_kwargs(json.load(file))
22/73: verbosity = config.pop("verbosity", None)
22/74: checkpoint = config.pop("checkpoint", {})
22/75: config["checkpoint"]
22/76: trainer, training_args = instantiate_trainer(**config)
22/77: checkpoint["resume_from_checkpoint"]
22/78: checkpoint["resume_from_checkpoint"] = None
22/79: checkpoint
22/80: trainer.model
22/81: trainer.model.qa_outputs
22/82: trainer.model.base_model
22/83: trainer.model
23/1:
import json
import torch
from tqdm import tqdm
from pathlib import Path
from datasets import load_from_disk, set_caching_enabled
from meerqat.train.trainee import MultiPassageBERT
from meerqat.data.loading import load_pretrained_in_kwargs
from meerqat.train.trainer import instantiate_trainer
from meerqat.train.trainer import get_checkpoint
from transformers.file_utils import WEIGHTS
24/1:
import json
import torch
from tqdm import tqdm
from pathlib import Path
from datasets import load_from_disk, set_caching_enabled
from meerqat.train.trainee import MultiPassageBERT
from meerqat.data.loading import load_pretrained_in_kwargs
from meerqat.train.trainer import instantiate_trainer
from meerqat.train.trainer import get_checkpoint
24/2: from transformers.file_utils import WEIGHTS_NAME
24/3: dataset = load_from_disk("data/viquae_dataset/test")
24/4: dataset = load_from_disk('data/viquae_dataset/')
24/5: kb = load_from_disk('data/viquae_passages/')
24/6: set_caching_enabled(False)
25/1:
import json
import torch
from tqdm import tqdm
from pathlib import Path
from datasets import load_from_disk, set_caching_enabled
from meerqat.train.trainee import MultiPassageBERT
from meerqat.data.loading import load_pretrained_in_kwargs
from meerqat.train.trainer import instantiate_trainer
from meerqat.train.trainer import get_checkpoint
25/2: from transformers.file_utils import WEIGHTS_NAME
25/3: set_caching_enabled(False)
25/4: dataset = load_from_disk('data/viquae_dataset/')
25/5: kb = load_from_disk('data/viquae_passages/')
25/6: config_path = Path("experiments/rc/viquae_6/test/config.json")
25/7:
with open(config_path, "r") as file: 
    config = load_pretrained_in_kwargs(json.load(file))
25/8:
verbosity = config.pop("verbosity", None)
checkpoint = config.pop("checkpoint", {})
trainer, training_args = instantiate_trainer(**config)
25/9:
device = trainer.args.device
resume_from_checkpoints = get_checkpoint(**checkpoint)
25/10: tqdm(resume_from_checkpoints, desc="Prediction")
25/11: WEIGHTS_NAME
25/12: state_dict_path = resume_from_checkpoint / WEIGHTS_NAME
25/13: resume_from_checkpoints
25/14: resume_from_checkpoints[0]
25/15: state_dict_path = resume_from_checkpoints[0] / WEIGHTS_NAME
25/16: state_dict_path
25/17: state_dict = torch.load(state_dict_path, map_location=device)
25/18: state_dict
25/19: trainer._load_state_dict_in_model(state_dict)
25/20: config_path = Path("experiments/rc/viquae_6/train/config.json")
25/21:
with open(config_path, "r") as file: 
    config = load_pretrained_in_kwargs(json.load(file))
25/22:
verbosity = config.pop("verbosity", None)
checkpoint = config.pop("checkpoint", {})
trainer, training_args = instantiate_trainer(**config)
25/23:
device = trainer.args.device
resume_from_checkpoints = get_checkpoint(**checkpoint)
25/24: state_dict_path = resume_from_checkpoints[0] / WEIGHTS_NAME
25/25: state_dict = torch.load(state_dict_path, map_location=device)
25/26: state_dict
25/27: trainer._load_state_dict_in_model(state_dict)
25/28: trainer.eval_dataset
25/29: trainer.train_dataset
26/1: from datasets import load_from_disk, set_caching_enabled
26/2: set_caching_enabled(False)
26/3: kb = load_from_disk('data/viquae_passages/')
26/4: dataset = load_from_disk('data/viquae_dataset/')
26/5: dataset
26/6: kb
26/7: item = kb[0]
26/8: item
26/9: kb[1]
26/10: dataset
26/11: dataset['train']
26/12: dataset['train'][0]
26/13: dataset['train'][0]['passage_scores']
26/14: dataset['train'][0]
26/15: dataset['train']
27/1: from datasets import load_from_disk, set_caching_enabled
27/2: dataset = load_from_disk("data/viquae_dataset/test")
27/3: dataset
27/4: kb = load_from_disk('data/viquae_passages/')
27/5: kb
27/6: dataset
27/7: dataset['search_scores']
27/8: import matplotlib.pyplot as plt
27/9: dataset
27/10: len(dataset['search_scores'])
27/11: plt.plot(dataset['search_scores'])
27/12: plt.show()
27/13: plt.savefig("passage_scores.png")
28/1: from datasets import load_from_disk, set_caching_enabled
28/2: import matplotlib.pyplot as plt
28/3: dataset = load_from_disk("data/viquae_dataset/test")
28/4: x  = range(0, len(dataset['search_scores']))
28/5: plt.bar(x, dataset['search_scores'])
28/6: x  = list(range(0, len(dataset['search_scores'])))
28/7: len(x)
28/8: plt.bar(x, dataset['search_scores'])
28/9: type(dataset['search_scores'])
28/10: len(dataset['search_scores'])
28/11: y = np.array(dataset['search_scores'])
28/12: import numpy as np
28/13: y = np.array(dataset['search_scores'])
28/14: x  = np.array(list(range(0, len(dataset['search_scores']))))
28/15: plt.bar(x,y)
28/16: x.shape, y.shape
28/17: dataset['search_scores']
28/18: plt.plot(y.flatten())
28/19: plt.savefig("passage_scores.png")
28/20: y.flatten().min()
28/21: y.flatten().max()
28/22: z = 1/(1 + np.exp(-y.flatten()))
28/23: z.shape
28/24: plt.plot(z)
28/25: plt.savefig("passage_scores_z.png")
28/26: plt.show()
28/27: plt.plot(z)
28/28: plt.show()
28/29: plt.savefig("passage_scores_z.png")
29/1: from datasets import load_from_disk, set_caching_enabled
29/2: set_caching_enabled(False)
29/3: kb = load_from_disk('data/viquae_passages')
29/4: dataset = load_from_disk('data/viquae_datasets')
29/5: dataset = load_from_disk('data/viquae_dataset')
29/6: kb
29/7: dataset
29/8: train_set = dataset['train']
29/9: train_set
29/10: item = train_set[0]
29/11: item
29/12: train_set
29/13: item['image']
29/14: item['input']
29/15: kb
29/16: train_set
29/17: item['image']
29/18: kb
29/19: wiki = load_from_disk('data/viquae_wiki')
29/20: wiki = load_from_disk('data/viquae_wikipedia')
29/21: wiki
29/22: non_humans = wiki['non_humans']
29/23: non_humans
30/1: from datasets import load_from_disk, set_caching_enabled
30/2: set_caching_enabled(False)
30/3: kb = load_from_disk('data/viquae_passages')
30/4: dataset = load_from_disk('data/viquae_datasets')
30/5: dataset = load_from_disk('data/viquae_dataset')
30/6: train_set = dataset['train']
30/7: wiki = load_from_disk('data/viquae_wiki')
30/8: wiki = load_from_disk('data/viquae_wikipedia')
30/9: non_humans = wiki['non_humans']
30/10: non_humans
30/11: item = non_humans[0]
30/12: item['document']
30/13: non_humans
30/14: item['image']
30/15: item['kilt_id']
30/16: item['passage_index']
30/17: item = non_humans[1]
30/18: item['passage_index']
30/19: non_humans
30/20: item['kilt_id']
30/21: item = train_set[0]
30/22: train_set
30/23: item['image']
30/24: item['image_hash']
30/25: kb
30/26: passage = kb[0]
30/27: passage
30/28: kb
30/29: non_humans
30/30: item['image_embedding']
30/31: item['image']
31/1: from datasets import load_from_disk, set_caching_enabled
31/2: set_caching_enabled(False)
31/3: kb = load_from_disk('data/viquae_passages')
32/1: from datasets import load_from_disk, set_caching_enabled
32/2: set_caching_enabled(False)
32/3: kb = load_from_disk('data/viquae_passages')
32/4: dataset = load_from_disk('data/viquae_dataset')
32/5: wiki = load_from_disk('data/viquae_wikipedia')
33/1:
import os, math
import os.path as osp
from copy import deepcopy
from functools import partial
from pprint import pprint
33/2:
import sacred
import torch
import torch.nn as nn
from sacred import SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds
from torch.backends import cudnn
from torch.optim import SGD, Adam, AdamW, lr_scheduler
# from visdom_logger import VisdomLogger
33/3:
from models.ingredient import model_ingredient, get_model
from utils import state_dict_to_cpu, num_of_trainable_params
from utils import pickle_load
from utils import BinaryCrossEntropyWithLogits
from utils.data.dataset_ingredient import data_ingredient, get_loaders
from utils.training import train_one_epoch, evaluate
33/4:
ex = sacred.Experiment('RRT Training', ingredients=[data_ingredient, model_ingredient])
# Filter backspaces and linefeeds
SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds
33/5: ex = sacred.Experiment('RRT Training', ingredients=[data_ingredient, model_ingredient])
33/6:
ex = sacred.Experiment('RRT Training', ingredients=[data_ingredient, model_ingredient], interactive=True)
# Filter backspaces and linefeeds
SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds
33/7:

@ex.config
def config():
    epochs = 15
    lr = 0.0001
    momentum = 0.
    nesterov = False
    weight_decay = 5e-4
    optim = 'adamw'
    scheduler = 'multistep'
    max_norm = 0.0
    seed = 0

    visdom_port = None
    visdom_freq = 100
    cpu = False  # Force training on CPU
    cudnn_flag = 'benchmark'
    temp_dir = osp.join('outputs', 'temp')

    no_bias_decay = False
    loss = 'bce'
    scheduler_tau = [16, 18]
    scheduler_gamma = 0.1

    resume = None
33/8:
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
33/9:
@ex.capture
def get_loss(loss):
    if loss == 'bce':
        return BinaryCrossEntropyWithLogits()
    else:
        raise Exception('Unsupported loss {}'.format(loss))
33/10:
@ex.automain
def main(epochs, cpu, cudnn_flag, visdom_port, visdom_freq, temp_dir, seed, no_bias_decay, max_norm, resume):
    device = torch.device('cuda:0' if torch.cuda.is_available() and not cpu else 'cpu')
    # callback = VisdomLogger(port=visdom_port) if visdom_port else None
    if cudnn_flag == 'deterministic':
        setattr(cudnn, cudnn_flag, True)

    torch.manual_seed(seed)
    loaders, recall_ks = get_loaders()

    torch.manual_seed(seed+1)
    model = get_model()
    if resume is not None:
        checkpoint = torch.load(resume, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state'], strict=True)
    print('# of trainable parameters: ', num_of_trainable_params(model))
    class_loss = get_loss()
    nn_inds_path = osp.join(loaders.query.dataset.data_dir, 'nn_inds_%s.pkl'%loaders.query.dataset.desc_name)
    cache_nn_inds = torch.from_numpy(pickle_load(nn_inds_path)).long()

    torch.manual_seed(seed+2)
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

    torch.manual_seed(seed+3)
    # setup partial function to simplify call
    eval_function = partial(evaluate, model=model, 
        cache_nn_inds=cache_nn_inds,
        recall=recall_ks, query_loader=loaders.query, gallery_loader=loaders.gallery)

    # setup best validation logger
    result = eval_function()
    # if callback is not None:
    #     callback.scalars(['l2', 'cosine'], 0, [metrics.recall['l2'][1], metrics.recall['cosine'][1]],
    #                      title='Val Recall@1')
    pprint(result)
    best_val = (0, result, deepcopy(model.state_dict()))

    # saving
    save_name = osp.join(temp_dir, '{}_{}.pt'.format(ex.current_run.config['model']['name'],
                                                         ex.current_run.config['dataset']['name']))
    os.makedirs(temp_dir, exist_ok=True)
    torch.manual_seed(seed+4)
    for epoch in range(epochs):
        if cudnn_flag == 'benchmark':
            setattr(cudnn, cudnn_flag, True)

        torch.cuda.empty_cache()
        # train_one_epoch(model=model, loader=loaders.train, class_loss=class_loss, optimizer=optimizer, scheduler=scheduler, epoch=epoch, callback=callback, freq=visdom_freq, ex=ex)
        # train_one_epoch(model=model, loader=loaders.train, class_loss=class_loss, optimizer=optimizer, scheduler=scheduler, max_norm=max_norm, epoch=epoch, callback=callback, freq=visdom_freq, ex=None)
        train_one_epoch(model=model, loader=loaders.train, class_loss=class_loss, optimizer=optimizer, scheduler=scheduler, max_norm=max_norm, epoch=epoch, freq=visdom_freq, ex=None)

        # validation
        if cudnn_flag == 'benchmark':
            setattr(cudnn, cudnn_flag, False)
        torch.cuda.empty_cache()
        result = eval_function()
        print('Validation [{:03d}]'.format(epoch)), pprint(result)
        ex.log_scalar('val.M_map', result['M_map'], step=epoch + 1)
        ex.log_scalar('val.H_map', result['H_map'], step=epoch + 1)

        if (result['M_map'] + result['H_map']) >= (best_val[1]['M_map'] + best_val[1]['H_map']):
            print('New best model in epoch %d.'%epoch)
            best_val = (epoch + 1, result, deepcopy(model.state_dict()))
            torch.save({'state': state_dict_to_cpu(best_val[2]), 'optim': optimizer.state_dict()}, save_name)

    # logging
    ex.info['metrics'] = best_val[1]
    ex.add_artifact(save_name)

    # if callback is not None:
    #     save_name = os.path.join(temp_dir, 'visdom_data.pt')
    #     callback.save(save_name)
    #     ex.add_artifact(save_name)

    return best_val[1]
33/11:
@ex.main
def main(epochs, cpu, cudnn_flag, visdom_port, visdom_freq, temp_dir, seed, no_bias_decay, max_norm, resume):
    device = torch.device('cuda:0' if torch.cuda.is_available() and not cpu else 'cpu')
    # callback = VisdomLogger(port=visdom_port) if visdom_port else None
    if cudnn_flag == 'deterministic':
        setattr(cudnn, cudnn_flag, True)

    torch.manual_seed(seed)
    loaders, recall_ks = get_loaders()

    torch.manual_seed(seed+1)
    model = get_model()
    if resume is not None:
        checkpoint = torch.load(resume, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state'], strict=True)
    print('# of trainable parameters: ', num_of_trainable_params(model))
    class_loss = get_loss()
    nn_inds_path = osp.join(loaders.query.dataset.data_dir, 'nn_inds_%s.pkl'%loaders.query.dataset.desc_name)
    cache_nn_inds = torch.from_numpy(pickle_load(nn_inds_path)).long()

    torch.manual_seed(seed+2)
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

    torch.manual_seed(seed+3)
    # setup partial function to simplify call
    eval_function = partial(evaluate, model=model, 
        cache_nn_inds=cache_nn_inds,
        recall=recall_ks, query_loader=loaders.query, gallery_loader=loaders.gallery)

    # setup best validation logger
    result = eval_function()
    # if callback is not None:
    #     callback.scalars(['l2', 'cosine'], 0, [metrics.recall['l2'][1], metrics.recall['cosine'][1]],
    #                      title='Val Recall@1')
    pprint(result)
    best_val = (0, result, deepcopy(model.state_dict()))

    # saving
    save_name = osp.join(temp_dir, '{}_{}.pt'.format(ex.current_run.config['model']['name'],
                                                         ex.current_run.config['dataset']['name']))
    os.makedirs(temp_dir, exist_ok=True)
    torch.manual_seed(seed+4)
    for epoch in range(epochs):
        if cudnn_flag == 'benchmark':
            setattr(cudnn, cudnn_flag, True)

        torch.cuda.empty_cache()
        # train_one_epoch(model=model, loader=loaders.train, class_loss=class_loss, optimizer=optimizer, scheduler=scheduler, epoch=epoch, callback=callback, freq=visdom_freq, ex=ex)
        # train_one_epoch(model=model, loader=loaders.train, class_loss=class_loss, optimizer=optimizer, scheduler=scheduler, max_norm=max_norm, epoch=epoch, callback=callback, freq=visdom_freq, ex=None)
        train_one_epoch(model=model, loader=loaders.train, class_loss=class_loss, optimizer=optimizer, scheduler=scheduler, max_norm=max_norm, epoch=epoch, freq=visdom_freq, ex=None)

        # validation
        if cudnn_flag == 'benchmark':
            setattr(cudnn, cudnn_flag, False)
        torch.cuda.empty_cache()
        result = eval_function()
        print('Validation [{:03d}]'.format(epoch)), pprint(result)
        ex.log_scalar('val.M_map', result['M_map'], step=epoch + 1)
        ex.log_scalar('val.H_map', result['H_map'], step=epoch + 1)

        if (result['M_map'] + result['H_map']) >= (best_val[1]['M_map'] + best_val[1]['H_map']):
            print('New best model in epoch %d.'%epoch)
            best_val = (epoch + 1, result, deepcopy(model.state_dict()))
            torch.save({'state': state_dict_to_cpu(best_val[2]), 'optim': optimizer.state_dict()}, save_name)

    # logging
    ex.info['metrics'] = best_val[1]
    ex.add_artifact(save_name)

    # if callback is not None:
    #     save_name = os.path.join(temp_dir, 'visdom_data.pt')
    #     callback.save(save_name)
    #     ex.add_artifact(save_name)

    return best_val[1]
33/12:
@ex.main
def main(epochs, cpu, cudnn_flag, visdom_port, visdom_freq, temp_dir, seed, no_bias_decay, max_norm, resume):
    device = torch.device('cuda:0' if torch.cuda.is_available() and not cpu else 'cpu')
    # callback = VisdomLogger(port=visdom_port) if visdom_port else None
    if cudnn_flag == 'deterministic':
        setattr(cudnn, cudnn_flag, True)

    torch.manual_seed(seed)
    loaders, recall_ks = get_loaders()

    torch.manual_seed(seed+1)
    model = get_model()
    if resume is not None:
        checkpoint = torch.load(resume, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state'], strict=True)
    print('# of trainable parameters: ', num_of_trainable_params(model))
    class_loss = get_loss()
    nn_inds_path = osp.join(loaders.query.dataset.data_dir, 'nn_inds_%s.pkl'%loaders.query.dataset.desc_name)
    cache_nn_inds = torch.from_numpy(pickle_load(nn_inds_path)).long()

    torch.manual_seed(seed+2)
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

    torch.manual_seed(seed+3)
    # setup partial function to simplify call
    eval_function = partial(evaluate, model=model, 
        cache_nn_inds=cache_nn_inds,
        recall=recall_ks, query_loader=loaders.query, gallery_loader=loaders.gallery)

    # setup best validation logger
    result = eval_function()
    # if callback is not None:
    #     callback.scalars(['l2', 'cosine'], 0, [metrics.recall['l2'][1], metrics.recall['cosine'][1]],
    #                      title='Val Recall@1')
    pprint(result)
    best_val = (0, result, deepcopy(model.state_dict()))

    # saving
    save_name = osp.join(temp_dir, '{}_{}.pt'.format(ex.current_run.config['model']['name'],
                                                         ex.current_run.config['dataset']['name']))
    os.makedirs(temp_dir, exist_ok=True)
    torch.manual_seed(seed+4)
    for epoch in range(epochs):
        if cudnn_flag == 'benchmark':
            setattr(cudnn, cudnn_flag, True)

        torch.cuda.empty_cache()
        # train_one_epoch(model=model, loader=loaders.train, class_loss=class_loss, optimizer=optimizer, scheduler=scheduler, epoch=epoch, callback=callback, freq=visdom_freq, ex=ex)
        # train_one_epoch(model=model, loader=loaders.train, class_loss=class_loss, optimizer=optimizer, scheduler=scheduler, max_norm=max_norm, epoch=epoch, callback=callback, freq=visdom_freq, ex=None)
        train_one_epoch(model=model, loader=loaders.train, class_loss=class_loss, optimizer=optimizer, scheduler=scheduler, max_norm=max_norm, epoch=epoch, freq=visdom_freq, ex=None)

        # validation
        if cudnn_flag == 'benchmark':
            setattr(cudnn, cudnn_flag, False)
        torch.cuda.empty_cache()
        result = eval_function()
        print('Validation [{:03d}]'.format(epoch)), pprint(result)
        ex.log_scalar('val.M_map', result['M_map'], step=epoch + 1)
        ex.log_scalar('val.H_map', result['H_map'], step=epoch + 1)

        if (result['M_map'] + result['H_map']) >= (best_val[1]['M_map'] + best_val[1]['H_map']):
            print('New best model in epoch %d.'%epoch)
            best_val = (epoch + 1, result, deepcopy(model.state_dict()))
            torch.save({'state': state_dict_to_cpu(best_val[2]), 'optim': optimizer.state_dict()}, save_name)

    # logging
    ex.info['metrics'] = best_val[1]
    ex.add_artifact(save_name)

    # if callback is not None:
    #     save_name = os.path.join(temp_dir, 'visdom_data.pt')
    #     callback.save(save_name)
    #     ex.add_artifact(save_name)

    return best_val[1]
34/1: from datasets import load_from_disk, set_caching_enabled
34/2: set_caching_enabled(False)
34/3: kb = load_from_disk('data/viquae_passages')
34/4: dataset = load_from_disk('data/viquae_dataset')
34/5: wiki = load_from_disk('data/viquae_wikipedia')
34/6: wiki
34/7: dataset
34/8: train_set = dataset['train']
34/9: train_sets
34/10: train_set
34/11: item = train_set[0]
34/12: item['original_question']
34/13: item['provenance_indices']
34/14: train_set
34/15: len(item['provenance_indices'])
34/16: len(item['search_indices'])
34/17: len(item['search_provenance_indices'])
34/18: len(item['search_irrelevant_indices'])
34/19: train_set
34/20: len(item['search_indices'])
34/21: kb
34/22: train_set
34/23: len(item['meta'])
35/1:
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
# from visdom_logger import VisdomLogger

from models.ingredient import model_ingredient, get_model
from utils import pickle_load
from utils.data.dataset_ingredient import data_ingredient, get_loaders
# from utils.training import evaluate_time as evaluate
from utils.training import evaluate

ex = sacred.Experiment('RRT Evaluation', ingredients=[data_ingredient, model_ingredient])
# Filter backspaces and linefeeds
SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds
36/1:
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
# from visdom_logger import VisdomLogger

from models.ingredient import model_ingredient, get_model
from utils import pickle_load
from utils.data.dataset_ingredient import data_ingredient, get_loaders
# from utils.training import evaluate_time as evaluate
from utils.training import evaluate

ex = sacred.Experiment('RRT Evaluation', ingredients=[data_ingredient, model_ingredient],interactive=True)
# Filter backspaces and linefeeds
SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds
36/2:
@ex.config
def config():
    visdom_port = None
    visdom_freq = 20
    cpu = False  # Force training on CPU
    cudnn_flag = 'benchmark'
    temp_dir = osp.join('logs', 'temp')
    resume = None
    seed = 0
36/3: device = torch.device('cuda:0' if torch.cuda.is_available() and not cpu else 'cpu')
36/4: device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
36/5: device
36/6: if cudnn_flag == 'deterministic':   setattr(cudnn, cudnn_flag, True)
36/7: if cudnn_flag == 'deterministic':   setattr(cudnn, cudnn_flag, True)
36/8:
if cudnn_flag == 'deterministic':
   setattr(cudnn, cudnn_flag, True)
36/9: torch.manual_seed(seed)
37/1:
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
# from visdom_logger import VisdomLogger

from models.ingredient import model_ingredient, get_model
from utils import pickle_load
from utils.data.dataset_ingredient import data_ingredient, get_loaders
# from utils.training import evaluate_time as evaluate
from utils.training import evaluate

ex = sacred.Experiment('RRT Evaluation', ingredients=[data_ingredient, model_ingredient],interactive=True)
# Filter backspaces and linefeeds
SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds
37/2: cpu = False
37/3: cudnn_flag = 'benchmark'
37/4: temp_dir = osp.join('logs', 'temp')
37/5: resume = None
37/6: seed = 0
37/7: device = torch.device('cuda:0' if torch.cuda.is_available() and not cpu else 'cpu')
37/8:
if cudnn_flag == 'deterministic':
    setattr(cudnn, cudnn_flag, True)
37/9: torch.manual_seed(seed)
37/10: loaders, recall_ks = get_loaders()
37/11: loaders, recall_ks = get_loaders("model.RTT", "file:///mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/roxford", 36, 36, 8, True, 'random', [1,5,10])
37/12: get_sets()
37/13: from utils.data.dataset_ingredient import get_sets
37/14: get_sets()
37/15:  (train_set, query_train_set), (query_set, gallery_set) = get_sets('r50_gldv1', '/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/oxford5k','/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/oxford5k','test_query.txt', ('test_query.txt', 'test_gallery.txt'), 'gnd_roxford5k.pkl', 500)
37/16: train_sampler = BatchSampler(RandomSampler(train_set), batch_size=batch_size, drop_last=False)
37/17: from torch.utils.data import DataLoader, RandomSampler, BatchSampler
37/18: train_sampler = BatchSampler(RandomSampler(train_set), batch_size=batch_size, drop_last=False)
37/19: train_sampler = BatchSampler(RandomSampler(train_set), batch_size=36, drop_last=False)
37/20: num_candidates = 100
37/21: recalls = [1, 5, 10]
37/22: test_batch_size = 36
37/23: batch_size      = 36
37/24: train_loader = DataLoader(train_set, batch_sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)
37/25: num_workers = 8
37/26: train_loader = DataLoader(train_set, batch_sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)
37/27: pin_memory  = True
37/28: recalls = [1, 5, 10]
37/29: train_loader = DataLoader(train_set, batch_sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)
37/30: query_train_loader = DataLoader(query_train_set, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory)
37/31: query_loader   = DataLoader(query_set, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory)
37/32: gallery_loader = DataLoader(gallery_set, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory)
37/33: loaders, recall_ks = MetricLoaders(train=train_loader, query_train=query_train_loader, query=query_loader, gallery=gallery_loader, num_classes=len(train_set.categories)), recalls
37/34:
class MetricLoaders(NamedTuple):
    train: DataLoader
    num_classes: int
    query: DataLoader
    query_train: DataLoader
    gallery: Optional[DataLoader] = None
37/35: from typing import NamedTuple, Optional
37/36:
class MetricLoaders(NamedTuple):
    train: DataLoader
    num_classes: int
    query: DataLoader
    query_train: DataLoader
    gallery: Optional[DataLoader] = None
37/37: loaders, recall_ks = MetricLoaders(train=train_loader, query_train=query_train_loader, query=query_loader, gallery=gallery_loader, num_classes=len(train_set.categories)), recalls
37/38: torch.manual_seed(seed)
37/39:
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
37/40:

from matcher import MatchERT
from sacred import Ingredient
37/41:

from models.matcher import MatchERT
from sacred import Ingredient
37/42:
def get_model(num_global_features, num_local_features, seq_len, dim_K, dim_feedforward, nhead, num_encoder_layers, dropout, activation, normalize_before):
    return MatchERT(d_global=num_global_features, d_model=num_local_features, seq_len=seq_len, d_K=dim_K, nhead=nhead, num_encoder_layers=num_encoder_layers, 
            dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, normalize_before=normalize_before)
37/43: model = get_model()
37/44: model = get_model(num_global_features,num_local_features,seq_len,dim_K,dim_feedforward,nhead,num_encoder_layers,dropout,activation,normalize_before)
37/45:
name = 'rrt'
seq_len = 1004
dim_K = 256
dim_feedforward = 1024
nhead = 4
num_encoder_layers = 6
dropout = 0.0 
activation = "relu"
normalize_before = False
37/46: model = get_model(num_global_features,num_local_features,seq_len,dim_K,dim_feedforward,nhead,num_encoder_layers,dropout,activation,normalize_before)
37/47: resume = None
37/48:
if resume is not None:
    checkpoint = torch.load(resume, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state'], strict=True)
37/49: model.to(device)
37/50: model.eval()
37/51: model.eval()
37/52: nn_inds_path = osp.join(loaders.query.dataset.data_dir, 'nn_inds_%s.pkl'%loaders.query.dataset.desc_name)
37/53: cache_nn_inds = torch.from_numpy(pickle_load(nn_inds_path)).long()
37/54: eval_function = partial(evaluate, model=model, cache_nn_inds=cache_nn_inds, recall=recall_ks, query_loader=loaders.query, gallery_loader=loaders.gallery)
37/55:  metrics = eval_function()
37/56: best_val = (0, metrics, deepcopy(model.state_dict()))
38/1:
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
# from visdom_logger import VisdomLogger

from models.ingredient import model_ingredient, get_model
from utils import pickle_load
from utils.data.dataset_ingredient import data_ingredient, get_loaders
# from utils.training import evaluate_time as evaluate
from utils.training import evaluate

ex = sacred.Experiment('RRT Evaluation', ingredients=[data_ingredient, model_ingredient],interactive=True)
# Filter backspaces and linefeeds
SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds
39/1:
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
# from visdom_logger import VisdomLogger

from models.ingredient import model_ingredient, get_model
from utils import pickle_load
from utils.data.dataset_ingredient import data_ingredient, get_loaders
# from utils.training import evaluate_time as evaluate
from utils.training import evaluate

ex = sacred.Experiment('RRT Evaluation', ingredients=[data_ingredient, model_ingredient],interactive=True)
# Filter backspaces and linefeeds
SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds
39/2: cpu = False
39/3: cudnn_flag = 'benchmark'
39/4:  temp_dir = osp.join('logs', 'temp')
39/5: resume = None
39/6: seed = 0
39/7: device = torch.device('cuda:0' if torch.cuda.is_available() and not cpu else 'cpu')
39/8:
if cudnn_flag == 'deterministic': 
     setattr(cudnn, cudnn_flag, True)
39/9: torch.manual_seed(seed)
39/10: from utils.data.dataset_ingredient import get_sets
39/11: (train_set, query_train_set), (query_set, gallery_set) = get_sets('r50_gldv1', '/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/oxford5k','/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/oxford5k','test_query.txt', ('test_query.txt', 'test_gallery.txt'), 'gnd_roxford5k.pkl', 500)
39/12: from torch.utils.data import DataLoader, RandomSampler, BatchSampler
39/13: train_sampler = BatchSampler(RandomSampler(train_set), batch_size=batch_size, drop_last=False)
39/14: num_candidates = 100
39/15: recalls = [1, 5, 10]
39/16: test_batch_size = 36
39/17: batch_size      = 36
39/18: train_sampler = BatchSampler(RandomSampler(train_set), batch_size=batch_size, drop_last=False)
39/19: num_workers = 8
39/20: pin_memory  = True
39/21: train_loader = DataLoader(train_set, batch_sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)
39/22: query_train_loader = DataLoader(query_train_set, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory)
39/23: query_loader   = DataLoader(query_set, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory)
39/24: gallery_loader = DataLoader(gallery_set, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory)
39/25: loaders, recall_ks = MetricLoaders(train=train_loader, query_train=query_train_loader, query=query_loader, gallery=gallery_loader, num_classes=len(train_set.categories)), recalls
39/26: from typing import NamedTuple, Optional
39/27:
class MetricLoaders(NamedTuple): 
     train: DataLoader 
     num_classes: int 
     query: DataLoader 
     query_train: DataLoader 
     gallery: Optional[DataLoader] = None
39/28: loaders, recall_ks = MetricLoaders(train=train_loader, query_train=query_train_loader, query=query_loader, gallery=gallery_loader, num_classes=len(train_set.categories)), recalls
39/29:
name = None 
    ...: num_global_features = 2048 
    ...: num_local_features = 128 
    ...: seq_len = None 
    ...: dim_K = None 
    ...: dim_feedforward = None 
    ...: nhead = None 
    ...: num_encoder_layers = None 
    ...: dropout = 0.0 
    ...: activation = "relu" normalize_before = False
39/30:
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
39/31: from models.matcher import MatchERT, from sacred import Ingredient
39/32:
from models.matcher import MatchERT 
from sacred import Ingredient
39/33:
def get_model(num_global_features, num_local_features, seq_len, dim_K, dim_feedforward, nhead, num_encoder_layers, dropout, activation, normalize_before): 
    return MatchERT(d_global=num_global_features, d_model=num_local_features, seq_len=seq_len, d_K=dim_K, nhead=nhead, num_encoder_layers=num_encoder_layers, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, normalize_before=normalize_before)
39/34:
name = 'rrt' 
seq_len = 1004 
dim_K = 256 
dim_feedforward = 1024 
nhead = 4 
num_encoder_layers = 6 
dropout = 0.0  
activation = "relu" 
normalize_before = False
39/35: model = get_model(num_global_features,num_local_features,seq_len,dim_K,dim_feedforward,nhead,num_encoder_layers,dropout,activation,normalize_before)
39/36: resume = None
39/37:
if resume is not None: 
     checkpoint = torch.load(resume, map_location=torch.device('cpu'))      model.load_state_dict(checkpoint['state'], strict=True)
39/38: if resume is not None:     checkpoint = torch.load(resume, map_location=torch.device('cpu'))      model.load_state_dict(checkpoint['state'], strict=True)
39/39:
if resume is not None: 
     checkpoint = torch.load(resume, map_location=torch.device('cpu'))
     model.load_state_dict(checkpoint['state'], strict=True)
39/40: model.to(device)
39/41: model.eval()
39/42: nn_inds_path = osp.join(loaders.query.dataset.data_dir, 'nn_inds_%s.pkl '%loaders.query.dataset.desc_name)
39/43: cache_nn_inds = torch.from_numpy(pickle_load(nn_inds_path)).long()
39/44: nn_inds_path = osp.join(loaders.query.dataset.data_dir, 'nn_inds_%s.pkl'%loaders.query.dataset.desc_name)
39/45: cache_nn_inds = torch.from_numpy(pickle_load(nn_inds_path)).long()
39/46: eval_function = partial(evaluate, model=model, cache_nn_inds=cache_nn_inds, recall=recall_ks, query_loader=loaders.query, gallery_loader=loaders.gallery)
39/47: metrics = eval_function()
39/48: model.predict
39/49: from utils import pickle_load, pickle_save, json_save, ReadSolution
39/50: gnd_file = "/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/oxford5k/gnd_roxford5k.pkl"
39/51: gnd = pickle_load(gnd_file)
39/52: query_names   = gnd['qimlist']
39/53: gallery_names = gnd['imlist']
39/54: query_names[0]
39/55: query_names[0]
39/56: query_names[:10]
39/57: gallery_names[:10]
39/58: type(gallery_names)
39/59: import numpy as np
39/60: gallery_names = np.array(gallery_names)
39/61: query_names = np.array(query_names)
39/62: gallery_names.shape, query_names.shape
39/63: (gallery_names == query_names).sum()
39/64: len(gnd)
39/65: type(gnd)
39/66: gnd
39/67: gnd.keys()
39/68: gnd[0]
39/69: gnd.get(0)
39/70: gnd.get(1)
39/71: gnd['gnd']
39/72: gnd['gnd'][0]
34/24: train_set
34/25: len(item['image'])
34/26: train_set
34/27: dataset
34/28: len(item['id'])
34/29: wiki
34/30: non_humans
34/31: humans
34/32: non_humans = wiki['non_humans']
34/33: non_humans
34/34: item = non_humans[0]
34/35: item['image']
39/73: gnd['gnd'][0]
39/74: gnd
39/75: gnd.keys()
39/76: type(gnd)
39/77: gnd.keys()
34/36: train_set
34/37: item = train_set[0]
34/38: item['image']
34/39: dev_set = dataset['validation']
34/40: item = dev_set[0]
34/41: item['image']
34/42: dataset
34/43: gnd = {"gnd": [], "imlist": [], "qimlist": []}
34/44: dev_set[0]['image']
34/45: dev_set['image']
34/46: gnd = {"gnd": [], "imlist": [], "qimlist": []}
34/47: gnd['imlist']
34/48: gnd['imlist'] = [1,2,3,4]
34/49: gnd['imlist']
34/50: gnd['qimlist'] = dev_set['image']
34/51: import os
34/52: path ="/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg"
34/53: files = os.lisdir(path)
34/54: files = os.listdir(path)
34/55: files
34/56: path ="/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/viquae_images/jpg/"
34/57: gnd['imlist'] = os.listdir(path)
34/58: gnd['qimlist'] = dev_set['image']
34/59: gnd_file
34/60: gnd_file = "/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/viquae_images/gnd_viquae.pkl"
34/61: pickle_save(gnd_file, gnd)
34/62: from utils import pickle_load, pickle_save, json_save, ReadSolution
34/63: os.getcwd()
34/64: os.chdir("/mnt/beegfs/home/smessoud/RerankingTransformer/RRT_GLD")
34/65: os.getcwd()
34/66: from utils import pickle_load, pickle_save, json_save, ReadSolution
34/67: pickle_save(gnd_file, gnd)
34/68: dev_set = dataset['test']
34/69: gnd_file = "/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/viquae_images/gnd_test_viquae.pkl"
34/70: gnd['qimlist'] = dev_set['image']
34/71: pickle_save(gnd_file, gnd)
39/78: query_names = np.array(query_names)
39/79: query_names[0]
39/80: x = query_names[0]
39/81: '_'.join(x.split('_')[:-1])
39/82: x = query_names[0]
39/83: '_'.join(x.split('_')[:-1])
39/84: gnd_file = "/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/viquae_images/gnd_dev_viquae.pkl"
39/85: gnd = pickle_load(gnd_file)
39/86: query_names   = gnd['qimlist']
39/87: gallery_names = gnd['imlist']
39/88: x = query_names[0]
39/89: '_'.join(x.split('_')[:-1])
39/90: query_names[0]
39/91:
for x in gallery_names:
    cat = '_'.join(x.split('_')[:-1])
    categories.append(cat)
39/92: categories = []
39/93:
for x in gallery_names:
    cat = '_'.join(x.split('_')[:-1])
    categories.append(cat)
39/94:
for x in query_names:
    cat = '_'.join(x.split('_')[:-1])
    categories.append(cat)
39/95: len(categories)
39/96: categories = sorted(list(set(categories)))
39/97: cat_to_label = dict(zip(categories, range(len(categories))))
39/98: len(cat_to_label)
39/99: cat_to_label
39/100: query_outs   = [','.join([ osp.join(prefix, x), str(cat_to_label['_'.join(x.split('_')[:-1])]) ]) for x in query_names]
39/101: prefix = 'jpg'
39/102: query_outs   = [','.join([ osp.join(prefix, x), str(cat_to_label['_'.join(x.split('_')[:-1])]) ]) for x in query_names]
39/103: gallery_outs = [','.join([ osp.join(prefix, x), str(cat_to_label['_'.join(x.split('_')[:-1])]) ]) for x in gallery_names]
39/104: query_outs
39/105: gallery_outs
39/106: len(gallery_outs), len(query_outs)
39/107: os
39/108: from tools.prepare_data import extract_resolution
39/109: from tools.prepare_data import _init_paths
39/110: from tools import _init_paths
39/111: from tools.prepare_data import extract_resolution
39/112: import tools
39/113: tools.prepare_data.extract_resolution
39/114:
def extract_resolution(data_dir, records, gnd=None):
    outs = []
    for i in range(len(records)):
        entry = records[i]
        name, label = entry.split(',')
        path = osp.join(data_dir, name)
        if gnd is not None:
            bbx = gnd['gnd'][i]['bbx']
            width  = int(bbx[2] - bbx[0] + 1)
            height = int(bbx[3] - bbx[1] + 1)
        else:
            try:
                img = Image.open(path)
            except Warning:
                print('corrupted image:', i, name)
            width, height = img.size
        line = ','.join([name, label, str(width), str(height)])
        outs.append(line)
        if i % 1000 == 0:
            print(i)
    return outs
39/115: data_dir = "/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/viquae_images/"
39/116: extract_resolution(data_dir, gallery_outs)
39/117: from PIL import Image
39/118: extract_resolution(data_dir, gallery_outs)
39/119: entry = gallery_outs[0]
39/120: entry
39/121: entry.split(',')
39/122:
for i in range(0,10):
    entry = records[i]
    name, label = entry.split(',')
    path = osp.join(data_dir, name)
39/123:
for i in range(0,10):
    entry = gallery_outs[i]
    name, label = entry.split(',')
    path = osp.join(data_dir, name)
39/124:
for i in range(0,5):
    entry = gallery_outs[i]
    name, label = entry.split(',')
    path = osp.join(data_dir, name)
39/125:
for i in range(0,2):
    entry = gallery_outs[i]
    name, label = entry.split(',')
    path = osp.join(data_dir, name)
39/126: entry = gallery_outs[4]
39/127: entry.split(',')
39/128: entry = gallery_outs[3]
39/129: entry.split(',')
39/130: '_'.join(x.split('_')[:-1])
39/131: entry
39/132: random.randint(start, stop)
39/133: import random
39/134: random.randint(start, stop)
39/135: random.randint(0, 100)
34/72: train_set
39/136: query_outs   = [','.join([ osp.join(prefix, x), str(cat_to_label['_'.join(x.split('_')[:-1])]) ]) for x in query_names]
39/137: gallery_outs = [','.join([ osp.join(prefix, x), str(cat_to_label['_'.join(x.split('_')[:-1])]) ]) for x in gallery_names]
39/138:
for i in range(0,10):
    entry = gallery_outs[i]
    name, label = entry.split(',')
    path = osp.join(data_dir, name)
39/139: random.randint(0, 100)
39/140:
for x in query_names:
    cat = '_'.join(x.split('_')[:-1])
    categories.append(cat)
39/141:
for x in gallery_names:
    cat = '_'.join(x.split('_')[:-1])
    categories.append(cat)
39/142:
for x in gallery_names:
    cat = random.randint(0, 100)
    categories.append(cat)
39/143:
for x in query_names:
    cat = random.randint(0, 100)
    categories.append(cat)
39/144: categories = sorted(list(set(categories)))
39/145: categories = []
39/146:
for x in gallery_names:
    cat = random.randint(0, 100)
    categories.append(cat)
39/147:
for x in query_names:
    cat = random.randint(0, 100)
    categories.append(cat)
39/148: categories = sorted(list(set(categories)))
39/149: cat_to_label = dict(zip(categories, range(len(categories))))
39/150: query_outs   = [','.join([ osp.join(prefix, x+'.jpg'), str(cat_to_label['_'.join(x.split('_')[:-1])]) ]) for x in query_names]
34/73: l
34/74: item['image']
34/75: item['face_prob']
34/76: wiki
39/151: query_outs   = [','.join([ osp.join(prefix, x), str(cat_to_label['_'.join(x.split('_')[:-1])]) ]) for x in query_names]
39/152: oxford_gnd_file = "/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/oxford5k/gnd_roxford5k.pkl"
39/153: oxford_gnd = pickle_load(oxford_gnd_file)
39/154: oxford_gnd.keys()
39/155: oxford_gnd.keys('imlist')
39/156: oxford_gnd['imlist'][0]
39/157: query_outs   = [','.join([ osp.join(prefix, x+'.jpg'), str(cat_to_label['_'.join(x.split('_')[:-1])]) ]) for x in query_names]
39/158: categories = []
39/159: ["1", "2", "3"]
39/160: mylist = ["1", "2", "3"]
39/161: mylist
39/162: mylist + 1
39/163: mylist + "1"
39/164: query_names = [i+"_1" for i in query_names]
39/165: gallery_names = [i+"_1" for i in gallery_names]
39/166: categories = []
39/167:
for x in gallery_names:
    cat = '_'.join(x.split('_')[:-1])
    categories.append(cat)
39/168:
for x in query_names:
    cat = '_'.join(x.split('_')[:-1])
    categories.append(cat)
39/169: categories = sorted(list(set(categories)))
39/170: cat_to_label = dict(zip(categories, range(len(categories))))
39/171: query_outs   = [','.join([ osp.join(prefix, x), str(cat_to_label['_'.join(x.split('_')[:-1])]) ]) for x in query_names]
39/172: gallery_outs = [','.join([ osp.join(prefix, x), str(cat_to_label['_'.join(x.split('_')[:-1])]) ]) for x in gallery_names]
39/173: test, index = query_outs, gallery_outs
39/174:
def extract_resolution(data_dir, records, gnd=None):
    outs = []
    for i in range(len(records)):
        entry = records[i]
        name, label = entry.split(',')
        path = osp.join(data_dir, name)
        if gnd is not None:
            bbx = gnd['gnd'][i]['bbx']
            width  = int(bbx[2] - bbx[0] + 1)
            height = int(bbx[3] - bbx[1] + 1)
        else:
            try:
                img = Image.open(path)
            except Warning:
                print('corrupted image:', i, name)
            width, height = img.size
        line = ','.join([name, label, str(width), str(height)])
        outs.append(line)
        if i % 1000 == 0:
            print(i)
    return outs
39/175: data_dir
39/176: test  = extract_resolution(data_dir, test)
39/177: query_outs   = [';'.join([ osp.join(prefix, x), str(cat_to_label['_'.join(x.split('_')[:-1])]) ]) for x in query_names]
39/178: gallery_outs = [';'.join([ osp.join(prefix, x), str(cat_to_label['_'.join(x.split('_')[:-1])]) ]) for x in gallery_names]
39/179:
def extract_resolution(data_dir, records, gnd=None):
    outs = []
    for i in range(len(records)):
        entry = records[i]
        name, label = entry.split(';')
        path = osp.join(data_dir, name)
        if gnd is not None:
            bbx = gnd['gnd'][i]['bbx']
            width  = int(bbx[2] - bbx[0] + 1)
            height = int(bbx[3] - bbx[1] + 1)
        else:
            try:
                img = Image.open(path)
            except Warning:
                print('corrupted image:', i, name)
            width, height = img.size
        line = ','.join([name, label, str(width), str(height)])
        outs.append(line)
        if i % 1000 == 0:
            print(i)
    return outs
39/180: test, index = query_outs, gallery_outs
39/181: test  = extract_resolution(data_dir, test)
39/182: test  = extract_resolution(data_dir, test)
39/183: len(test)
39/184: test, index = query_outs, gallery_outs
39/185: test  = extract_resolution(data_dir, test)
39/186: test  = extract_resolution(data_dir, test)
39/187: index  = extract_resolution(data_dir, index)
39/188:
for x in range(0,100):
    entry = records[i]
    name, label = entry.split(';')
39/189:
for x in range(0,100):
    entry = index[i]
    name, label = entry.split(';')
39/190:
for x in range(0,500):
    entry = index[i]
    name, label = entry.split(';')
39/191:
for x in range(500,100):
    entry = index[i]
    name, label = entry.split(';')
39/192:
for x in range(500,1000):
    entry = index[i]
    name, label = entry.split(';')
39/193: index  = extract_resolution(data_dir, index)
39/194:
for x in range(len(index)):
    entry = index[i]
    name, label = entry.split(';')
39/195: test, index = query_outs, gallery_outs
39/196:
for x in range(len(index)):
    entry = index[i]
    name, label = entry.split(';')
39/197: index  = extract_resolution(data_dir, index)
39/198:
for i in range(len(index)):
    entry = index[i]
    name, label = entry.split(';')
39/199:
for i in range(100):
    entry = index[i]
    name, label = entry.split(';')
39/200:
for i in range(500):
    entry = index[i]
    name, label = entry.split(';')
39/201:
for i in range(1000):
    entry = index[i]
    name, label = entry.split(';')
39/202:
for i in range(800):
    entry = index[i]
    name, label = entry.split(';')
39/203:
for i in range(700):
    entry = index[i]
    name, label = entry.split(';')
39/204:
for i in range(750):
    entry = index[i]
    name, label = entry.split(';')
39/205:
for i in range(780):
    entry = index[i]
    name, label = entry.split(';')
39/206:
for i in range(750,780):
    entry = index[i]
    name, label = entry.split(';')
39/207:
for i in range(760,780):
    entry = index[i]
    name, label = entry.split(';')
39/208:
for i in range(770,780):
    entry = index[i]
    name, label = entry.split(';')
39/209:
for i in range(770,780):
    print(i)
    entry = index[i]
    name, label = entry.split(';')
39/210: entry = index[771]
39/211: entry = index[771].split(";")
39/212: entry = index[772].split(";")
39/213: index[772].split(";")
39/214: index[771].split(";")
39/215: index[771]
39/216: test, index = query_outs, gallery_outs
39/217:
def extract_resolution(data_dir, records, gnd=None):
    outs = []
    for i in range(len(records)):
        entry = records[i]
        name, label = entry.split(';;')
        path = osp.join(data_dir, name)
        if gnd is not None:
            bbx = gnd['gnd'][i]['bbx']
            width  = int(bbx[2] - bbx[0] + 1)
            height = int(bbx[3] - bbx[1] + 1)
        else:
            try:
                img = Image.open(path)
            except Warning:
                print('corrupted image:', i, name)
            width, height = img.size
        line = ','.join([name, label, str(width), str(height)])
        outs.append(line)
        if i % 1000 == 0:
            print(i)
    return outs
39/218: gallery_outs = [';;'.join([ osp.join(prefix, x), str(cat_to_label['_'.join(x.split('_')[:-1])]) ]) for x in gallery_names]
39/219: query_outs   = [';;'.join([ osp.join(prefix, x), str(cat_to_label['_'.join(x.split('_')[:-1])]) ]) for x in query_names]
39/220: test, index = query_outs, gallery_outs
39/221: index  = extract_resolution(data_dir, index)
39/222: test  = extract_resolution(data_dir, test)
40/1: from utils import pickle_load, pickle_save, json_save, ReadSolution
41/1: from utils import pickle_load, pickle_save, json_save, ReadSolution
41/2: gnd_file = "/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/viquae_images/gnd_dev_viquae.pkl"
41/3: gnd = pickle_load(gnd_file)
41/4: gnd['imlist']
41/5: gnd['qimlist']
41/6: query_names   = gnd['qimlist']
41/7: gallery_names = [i+"_1" for i in gallery_names]
41/8: query_names   = gnd['qimlist']
41/9: gallery_names = gnd['imlist']
41/10: gallery_names = [i+"_1" for i in gallery_names]
41/11: gallery_names = [i+"_1" for i in gallery_names]
41/12:
categories = []
for x in query_names:
    cat = '_'.join(x.split('_')[:-1])
    categories.append(cat)
for x in gallery_names:
    cat = '_'.join(x.split('_')[:-1])
    categories.append(cat)
41/13:
categories = sorted(list(set(categories)))
cat_to_label = dict(zip(categories, range(len(categories))))
41/14: query_outs   = [';;'.join([ osp.join(prefix, x), str(cat_to_label['_'.join(x.split('_')[:-1])]) ]) for x in query_names]
41/15: import os.path as osp
41/16: query_outs   = [';;'.join([ osp.join(prefix, x), str(cat_to_label['_'.join(x.split('_')[:-1])]) ]) for x in query_names]
41/17: prefix = 'jpg'
41/18: query_outs   = [';;'.join([ osp.join(prefix, x), str(cat_to_label['_'.join(x.split('_')[:-1])]) ]) for x in query_names]
41/19: gallery_outs = [','.join([ osp.join(prefix, x+'.jpg'), str(cat_to_label['_'.join(x.split('_')[:-1])]) ]) for x in gallery_names]
41/20: test, index = query_outs, gallery_outs
41/21: test  = extract_resolution(data_dir, test, gnd)
41/22:
def extract_resolution(data_dir, records, gnd=None):
    outs = []
    for i in range(len(records)):
        entry = records[i]
        name, label = entry.split(';;')
        path = osp.join(data_dir, name)
        if gnd is not None:
            bbx = gnd['gnd'][i]['bbx']
            width  = int(bbx[2] - bbx[0] + 1)
            height = int(bbx[3] - bbx[1] + 1)
        else:
            try:
                img = Image.open(path)
            except Warning:
                print('corrupted image:', i, name)
            width, height = img.size
        line = ','.join([name, label, str(width), str(height)])
        outs.append(line)
        if i % 1000 == 0:
            print(i)
    return outs
41/23: test  = extract_resolution(data_dir, test, gnd)
41/24: data_dir = "/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/viquae_images/"
41/25: test  = extract_resolution(data_dir, test, gnd)
41/26: test  = extract_resolution(data_dir, test)
41/27: from PIL import Image
41/28: test  = extract_resolution(data_dir, test)
41/29: index = extract_resolution(data_dir, index)
41/30:
def extract_resolution(data_dir, records, gnd=None):
    outs = []
    for i in range(len(records)):
        entry = records[i]
        name, label = entry.split(';;')
        path = osp.join(data_dir, name)
        if gnd is not None:
            bbx = gnd['gnd'][i]['bbx']
            width  = int(bbx[2] - bbx[0] + 1)
            height = int(bbx[3] - bbx[1] + 1)
        else:
            try:
                img = Image.open(path)
            except Warning:
                print('corrupted image:', i, name)
            width, height = img.size
        line = ';;'.join([name, label, str(width), str(height)])
        outs.append(line)
        if i % 1000 == 0:
            print(i)
    return outs
41/31: test, index = query_outs, gallery_outs
41/32: index = extract_resolution(data_dir, index)
41/33: test, index = query_outs, gallery_outs
41/34: test  = extract_resolution(data_dir, test)
41/35: index = extract_resolution(data_dir, index)
41/36: gallery_outs = [';;'.join([ osp.join(prefix, x+'.jpg'), str(cat_to_label['_'.join(x.split('_')[:-1])]) ]) for x in gallery_names]
41/37: test, index = query_outs, gallery_outs
41/38: test  = extract_resolution(data_dir, test)
41/39: index = extract_resolution(data_dir, index)
41/40: query_outs   = [';;'.join([ osp.join(prefix, x), str(cat_to_label['_'.join(x.split('_')[:-1])]) ]) for x in query_names]
41/41: gallery_outs = [';;'.join([ osp.join(prefix, x), str(cat_to_label['_'.join(x.split('_')[:-1])]) ]) for x in gallery_names]
41/42: test, index = query_outs, gallery_outs
41/43: test  = extract_resolution(data_dir, test)
41/44: index = extract_resolution(data_dir, index)
41/45: gallery_outs = [';;'.join([ osp.join(prefix, x), str(cat_to_label['_'.join(x.split('_')[:-1])]) ]) for x in gallery_names]
41/46: test, index = query_outs, gallery_outs
41/47: index = extract_resolution(data_dir, index)
41/48: query_names   = gnd['qimlist']
41/49: gallery_names = gnd['imlist']
41/50:
categories = []
for x in query_names:
    cat = '_'.join(x.split('_')[:-1])
    categories.append(cat)
for x in gallery_names:
    cat = '_'.join(x.split('_')[:-1])
    categories.append(cat)
41/51:
categories = sorted(list(set(categories)))
cat_to_label = dict(zip(categories, range(len(categories))))
41/52: gallery_outs = [';;'.join([ osp.join(prefix, x), str(cat_to_label['_'.join(x.split('_')[:-1])]) ]) for x in gallery_names]
41/53: query_outs   = [';;'.join([ osp.join(prefix, x), str(cat_to_label['_'.join(x.split('_')[:-1])]) ]) for x in query_names]
41/54: test, index = query_outs, gallery_outs
41/55: index = extract_resolution(data_dir, index)
41/56: test  = extract_resolution(data_dir, test)
41/57:
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from delf import delf_config_pb2
from delf import datum_io
from delf import feature_io
from delf import utils
from delf.python.detect_to_retrieve import dataset
from delf import extractor
41/58:
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from delf import delf_config_pb2
from delf import datum_io
from delf import feature_io
from delf import utils
from delf.python.datasets.revisited_op import dataset
from delf import extractor
41/59:
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'delf_config_path', '/tmp/delf_config_example.pbtxt',
    'Path to DelfConfig proto text file with configuration to be used for DELG '
    'extraction. Local features are extracted if use_local_features is True; '
    'global features are extracted if use_global_features is True.')
flags.DEFINE_string(
    'dataset_file_path', '/tmp/gnd_roxford5k.mat',
    'Dataset file for Revisited Oxford or Paris dataset, in .mat format.')
flags.DEFINE_string(
    'images_dir', '/tmp/images',
    'Directory where dataset images are located, all in .jpg format.')
flags.DEFINE_string(
    'output_features_dir', '/tmp/features',
    "Directory where DELG features will be written to. Each image's features "
    'will be written to files with same name but different extension: the '
    'global feature is written to a file with extension .delg_global and the '
    'local features are written to a file with extension .delg_local.')
41/60:
# Extensions.
_DELG_GLOBAL_EXTENSION = '.delg_global'
_DELG_LOCAL_EXTENSION = '.delg_local'

# Pace to report extraction log.
_STATUS_CHECK_ITERATIONS = 100
41/61:
def read_file(filename):
    with open(filename) as f:
        lines = f.read().splitlines()
    return lines
41/62: print('Reading list of images from dataset file...')
41/63: FLAGS.dataset_file_path
41/64: dataset_file_path = "data/viquae_images/test_gallery.txt"
41/65: dataset_file_path = "/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/viquae_images/test_gallery.txt"
41/66: image_list = read_file(dataset_file_path)
41/67: image_list
41/68: num_images = len(image_list)
41/69: num_images
41/70: print('done! Found %d images' % num_images)
41/71: config = delf_config_pb2.DelfConfig()
41/72: dataset_file_path = "/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/viquae_images/test_gallery.txt"
41/73: delf_config_path = "/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/r50delg_gld_config.pbtxt"
41/74:
with tf.io.gfile.GFile(FLAGS.delf_config_path, 'r') as f:
  text_format.Parse(f.read(), config)
41/75:
with tf.io.gfile.GFile(delf_config_path, 'r') as f:
  text_format.Parse(f.read(), config)
41/76:
if not tf.io.gfile.exists(FLAGS.output_features_dir):
  tf.io.gfile.makedirs(FLAGS.output_features_dir)
41/77: delf_config_path = "/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/r50delg_gld_config.pbtxt"
41/78: output_features_dir = "/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/viquae_images/delg_r50_gldv1"
41/79:
if not tf.io.gfile.exists(output_features_dir):
  tf.io.gfile.makedirs(output_features_dir)
41/80: extractor_fn = extractor.MakeExtractor(config)
41/81: import os
41/82: os.chdir("/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg")
41/83: extractor_fn = extractor.MakeExtractor(config)
41/84:
start = time.time()
missing_images = []
41/85:
for i in range(num_images):
    if i == 0:
      print('Starting to extract features...')
    elif i % _STATUS_CHECK_ITERATIONS == 0:
      elapsed = (time.time() - start)
      print('Processing image %d out of %d, last %d '
            'images took %f seconds' %
            (i, num_images, _STATUS_CHECK_ITERATIONS, elapsed))
      start = time.time()
    line = image_list[i]
    image_path, image_label, image_width, image_height = line.split(',')
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    input_image_filename = os.path.join(FLAGS.images_dir, image_path)
    if not os.path.exists(input_image_filename):
        missing_images.append(image_name)
        continue
    
    # Compose output file name and decide if image should be skipped.
    should_skip_global = True
    should_skip_local = True
    if config.use_global_features:
      output_global_feature_filename = os.path.join(FLAGS.output_features_dir, image_name + _DELG_GLOBAL_EXTENSION)
      if not tf.io.gfile.exists(output_global_feature_filename):
        should_skip_global = False
    if config.use_local_features:
      output_local_feature_filename = os.path.join(FLAGS.output_features_dir, image_name + _DELG_LOCAL_EXTENSION)
      if not tf.io.gfile.exists(output_local_feature_filename):
        should_skip_local = False
    if should_skip_global and should_skip_local:
      print('Skipping %s' % image_name)
      continue

    pil_im = utils.RgbLoader(input_image_filename)
    resize_factor = 1.0
    # if FLAGS.image_set == 'query':
    #   # Crop query image according to bounding box.
    #   original_image_size = max(pil_im.size)
    #   bbox = [int(round(b)) for b in ground_truth[i]['bbx']]
    #   pil_im = pil_im.crop(bbox)
    #   cropped_image_size = max(pil_im.size)
    #   resize_factor = cropped_image_size / original_image_size
    im = np.array(pil_im)

    # Extract and save features.
    extracted_features = extractor_fn(im, resize_factor)
    if config.use_global_features:
      global_descriptor = extracted_features['global_descriptor']
      datum_io.WriteToFile(global_descriptor, output_global_feature_filename)
    if config.use_local_features:
      locations = extracted_features['local_features']['locations']
      descriptors = extracted_features['local_features']['descriptors']
      feature_scales = extracted_features['local_features']['scales']
      attention = extracted_features['local_features']['attention']
      feature_io.WriteToFile(output_local_feature_filename, locations,
                             feature_scales, descriptors, attention)
41/86:
for i in range(num_images):
    if i == 0:
      print('Starting to extract features...')
    elif i % _STATUS_CHECK_ITERATIONS == 0:
      elapsed = (time.time() - start)
      print('Processing image %d out of %d, last %d '
            'images took %f seconds' %
            (i, num_images, _STATUS_CHECK_ITERATIONS, elapsed))
      start = time.time()
    line = image_list[i]
    image_path, image_label, image_width, image_height = line.split(';;')
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    input_image_filename = os.path.join(FLAGS.images_dir, image_path)
    if not os.path.exists(input_image_filename):
        missing_images.append(image_name)
        continue
    
    # Compose output file name and decide if image should be skipped.
    should_skip_global = True
    should_skip_local = True
    if config.use_global_features:
      output_global_feature_filename = os.path.join(FLAGS.output_features_dir, image_name + _DELG_GLOBAL_EXTENSION)
      if not tf.io.gfile.exists(output_global_feature_filename):
        should_skip_global = False
    if config.use_local_features:
      output_local_feature_filename = os.path.join(FLAGS.output_features_dir, image_name + _DELG_LOCAL_EXTENSION)
      if not tf.io.gfile.exists(output_local_feature_filename):
        should_skip_local = False
    if should_skip_global and should_skip_local:
      print('Skipping %s' % image_name)
      continue

    pil_im = utils.RgbLoader(input_image_filename)
    resize_factor = 1.0
    # if FLAGS.image_set == 'query':
    #   # Crop query image according to bounding box.
    #   original_image_size = max(pil_im.size)
    #   bbox = [int(round(b)) for b in ground_truth[i]['bbx']]
    #   pil_im = pil_im.crop(bbox)
    #   cropped_image_size = max(pil_im.size)
    #   resize_factor = cropped_image_size / original_image_size
    im = np.array(pil_im)

    # Extract and save features.
    extracted_features = extractor_fn(im, resize_factor)
    if config.use_global_features:
      global_descriptor = extracted_features['global_descriptor']
      datum_io.WriteToFile(global_descriptor, output_global_feature_filename)
    if config.use_local_features:
      locations = extracted_features['local_features']['locations']
      descriptors = extracted_features['local_features']['descriptors']
      feature_scales = extracted_features['local_features']['scales']
      attention = extracted_features['local_features']['attention']
      feature_io.WriteToFile(output_local_feature_filename, locations,
                             feature_scales, descriptors, attention)
41/87:
for i in range(num_images):
    if i == 0:
      print('Starting to extract features...')
    elif i % _STATUS_CHECK_ITERATIONS == 0:
      elapsed = (time.time() - start)
      print('Processing image %d out of %d, last %d '
            'images took %f seconds' %
            (i, num_images, _STATUS_CHECK_ITERATIONS, elapsed))
      start = time.time()
    line = image_list[i]
    image_path, image_label, image_width, image_height = line.split(';;')
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    input_image_filename = os.path.join(images_dir, image_path)
    if not os.path.exists(input_image_filename):
        missing_images.append(image_name)
        continue
    
    # Compose output file name and decide if image should be skipped.
    should_skip_global = True
    should_skip_local = True
    if config.use_global_features:
      output_global_feature_filename = os.path.join(output_features_dir, image_name + _DELG_GLOBAL_EXTENSION)
      if not tf.io.gfile.exists(output_global_feature_filename):
        should_skip_global = False
    if config.use_local_features:
      output_local_feature_filename = os.path.join(FLAGS.output_features_dir, image_name + _DELG_LOCAL_EXTENSION)
      if not tf.io.gfile.exists(output_local_feature_filename):
        should_skip_local = False
    if should_skip_global and should_skip_local:
      print('Skipping %s' % image_name)
      continue

    pil_im = utils.RgbLoader(input_image_filename)
    resize_factor = 1.0
    # if FLAGS.image_set == 'query':
    #   # Crop query image according to bounding box.
    #   original_image_size = max(pil_im.size)
    #   bbox = [int(round(b)) for b in ground_truth[i]['bbx']]
    #   pil_im = pil_im.crop(bbox)
    #   cropped_image_size = max(pil_im.size)
    #   resize_factor = cropped_image_size / original_image_size
    im = np.array(pil_im)

    # Extract and save features.
    extracted_features = extractor_fn(im, resize_factor)
    if config.use_global_features:
      global_descriptor = extracted_features['global_descriptor']
      datum_io.WriteToFile(global_descriptor, output_global_feature_filename)
    if config.use_local_features:
      locations = extracted_features['local_features']['locations']
      descriptors = extracted_features['local_features']['descriptors']
      feature_scales = extracted_features['local_features']['scales']
      attention = extracted_features['local_features']['attention']
      feature_io.WriteToFile(output_local_feature_filename, locations,
                             feature_scales, descriptors, attention)
41/88: output_features_dir = "/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/viquae_images/delg_r50_gldv1"
41/89: images_dir = "/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/viquae_images/jpg"
41/90:
for i in range(num_images):
    if i == 0:
      print('Starting to extract features...')
    elif i % _STATUS_CHECK_ITERATIONS == 0:
      elapsed = (time.time() - start)
      print('Processing image %d out of %d, last %d '
            'images took %f seconds' %
            (i, num_images, _STATUS_CHECK_ITERATIONS, elapsed))
      start = time.time()
    line = image_list[i]
    image_path, image_label, image_width, image_height = line.split(';;')
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    input_image_filename = os.path.join(images_dir, image_path)
    if not os.path.exists(input_image_filename):
        missing_images.append(image_name)
        continue
    
    # Compose output file name and decide if image should be skipped.
    should_skip_global = True
    should_skip_local = True
    if config.use_global_features:
      output_global_feature_filename = os.path.join(output_features_dir, image_name + _DELG_GLOBAL_EXTENSION)
      if not tf.io.gfile.exists(output_global_feature_filename):
        should_skip_global = False
    if config.use_local_features:
      output_local_feature_filename = os.path.join(FLAGS.output_features_dir, image_name + _DELG_LOCAL_EXTENSION)
      if not tf.io.gfile.exists(output_local_feature_filename):
        should_skip_local = False
    if should_skip_global and should_skip_local:
      print('Skipping %s' % image_name)
      continue

    pil_im = utils.RgbLoader(input_image_filename)
    resize_factor = 1.0
    # if FLAGS.image_set == 'query':
    #   # Crop query image according to bounding box.
    #   original_image_size = max(pil_im.size)
    #   bbox = [int(round(b)) for b in ground_truth[i]['bbx']]
    #   pil_im = pil_im.crop(bbox)
    #   cropped_image_size = max(pil_im.size)
    #   resize_factor = cropped_image_size / original_image_size
    im = np.array(pil_im)

    # Extract and save features.
    extracted_features = extractor_fn(im, resize_factor)
    if config.use_global_features:
      global_descriptor = extracted_features['global_descriptor']
      datum_io.WriteToFile(global_descriptor, output_global_feature_filename)
    if config.use_local_features:
      locations = extracted_features['local_features']['locations']
      descriptors = extracted_features['local_features']['descriptors']
      feature_scales = extracted_features['local_features']['scales']
      attention = extracted_features['local_features']['attention']
      feature_io.WriteToFile(output_local_feature_filename, locations,
                             feature_scales, descriptors, attention)
41/91:
for i in range(num_images):
    if i == 0:
      print('Starting to extract features...')
    elif i % _STATUS_CHECK_ITERATIONS == 0:
      elapsed = (time.time() - start)
      print('Processing image %d out of %d, last %d '
            'images took %f seconds' %
            (i, num_images, _STATUS_CHECK_ITERATIONS, elapsed))
      start = time.time()
    line = image_list[i]
    image_path, image_label, image_width, image_height = line.split(';;')
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    input_image_filename = os.path.join(images_dir, image_path)
    if not os.path.exists(input_image_filename):
        missing_images.append(image_name)
        continue
    
    # Compose output file name and decide if image should be skipped.
    should_skip_global = True
    should_skip_local = True
    if config.use_global_features:
      output_global_feature_filename = os.path.join(output_features_dir, image_name + _DELG_GLOBAL_EXTENSION)
      if not tf.io.gfile.exists(output_global_feature_filename):
        should_skip_global = False
    if config.use_local_features:
      output_local_feature_filename = os.path.join(FLAGS.output_features_dir, image_name + _DELG_LOCAL_EXTENSION)
      if not tf.io.gfile.exists(output_local_feature_filename):
        should_skip_local = False
    if should_skip_global and should_skip_local:
      print('Skipping %s' % image_name)
      continue

    pil_im = utils.RgbLoader(input_image_filename)
    resize_factor = 1.0
    # if FLAGS.image_set == 'query':
    #   # Crop query image according to bounding box.
    #   original_image_size = max(pil_im.size)
    #   bbox = [int(round(b)) for b in ground_truth[i]['bbx']]
    #   pil_im = pil_im.crop(bbox)
    #   cropped_image_size = max(pil_im.size)
    #   resize_factor = cropped_image_size / original_image_size
    im = np.array(pil_im)

    # Extract and save features.
    extracted_features = extractor_fn(im, resize_factor)
    if config.use_global_features:
      global_descriptor = extracted_features['global_descriptor']
      datum_io.WriteToFile(global_descriptor, output_global_feature_filename)
    if config.use_local_features:
      locations = extracted_features['local_features']['locations']
      descriptors = extracted_features['local_features']['descriptors']
      feature_scales = extracted_features['local_features']['scales']
      attention = extracted_features['local_features']['attention']
      feature_io.WriteToFile(output_local_feature_filename, locations,
                             feature_scales, descriptors, attention)
41/92: output_features_dir
41/93: locations
41/94:
for i in range(num_images):
if i == 0:
  print('Starting to extract features...')
elif i % _STATUS_CHECK_ITERATIONS == 0:
  elapsed = (time.time() - start)
  print('Processing image %d out of %d, last %d '
        'images took %f seconds' %
        (i, num_images, _STATUS_CHECK_ITERATIONS, elapsed))
  start = time.time()
line = image_list[i]
image_path, image_label, image_width, image_height = line.split(',')
image_name = os.path.splitext(os.path.basename(image_path))[0]
input_image_filename = os.path.join(images_dir, image_path)
if not os.path.exists(input_image_filename):
    missing_images.append(image_name)
    continue

# Compose output file name and decide if image should be skipped.
should_skip_global = True
should_skip_local = True
if config.use_global_features:
  output_global_feature_filename = os.path.join(output_features_dir, image_name + _DELG_GLOBAL_EXTENSION)
  if not tf.io.gfile.exists(output_global_feature_filename):
    should_skip_global = False
if config.use_local_features:
  output_local_feature_filename = os.path.join(output_features_dir, image_name + _DELG_LOCAL_EXTENSION)
  if not tf.io.gfile.exists(output_local_feature_filename):
    should_skip_local = False
if should_skip_global and should_skip_local:
  print('Skipping %s' % image_name)
  continue

pil_im = utils.RgbLoader(input_image_filename)
resize_factor = 1.0
# if image_set == 'query':
#   # Crop query image according to bounding box.
#   original_image_size = max(pil_im.size)
#   bbox = [int(round(b)) for b in ground_truth[i]['bbx']]
#   pil_im = pil_im.crop(bbox)
#   cropped_image_size = max(pil_im.size)
#   resize_factor = cropped_image_size / original_image_size
im = np.array(pil_im)

# Extract and save features.
extracted_features = extractor_fn(im, resize_factor)
if config.use_global_features:
  global_descriptor = extracted_features['global_descriptor']
  datum_io.WriteToFile(global_descriptor, output_global_feature_filename)
if config.use_local_features:
  locations = extracted_features['local_features']['locations']
  descriptors = extracted_features['local_features']['descriptors']
  feature_scales = extracted_features['local_features']['scales']
  attention = extracted_features['local_features']['attention']
  feature_io.WriteToFile(output_local_feature_filename, locations,
                         feature_scales, descriptors, attention)
41/95:
  for i in range(num_images):
    if i == 0:
      print('Starting to extract features...')
    elif i % _STATUS_CHECK_ITERATIONS == 0:
      elapsed = (time.time() - start)
      print('Processing image %d out of %d, last %d '
            'images took %f seconds' %
            (i, num_images, _STATUS_CHECK_ITERATIONS, elapsed))
      start = time.time()
    line = image_list[i]
    image_path, image_label, image_width, image_height = line.split(',')
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    input_image_filename = os.path.join(images_dir, image_path)
    if not os.path.exists(input_image_filename):
        missing_images.append(image_name)
        continue
    
    # Compose output file name and decide if image should be skipped.
    should_skip_global = True
    should_skip_local = True
    if config.use_global_features:
      output_global_feature_filename = os.path.join(output_features_dir, image_name + _DELG_GLOBAL_EXTENSION)
      if not tf.io.gfile.exists(output_global_feature_filename):
        should_skip_global = False
    if config.use_local_features:
      output_local_feature_filename = os.path.join(output_features_dir, image_name + _DELG_LOCAL_EXTENSION)
      if not tf.io.gfile.exists(output_local_feature_filename):
        should_skip_local = False
    if should_skip_global and should_skip_local:
      print('Skipping %s' % image_name)
      continue

    pil_im = utils.RgbLoader(input_image_filename)
    resize_factor = 1.0
    # if image_set == 'query':
    #   # Crop query image according to bounding box.
    #   original_image_size = max(pil_im.size)
    #   bbox = [int(round(b)) for b in ground_truth[i]['bbx']]
    #   pil_im = pil_im.crop(bbox)
    #   cropped_image_size = max(pil_im.size)
    #   resize_factor = cropped_image_size / original_image_size
    im = np.array(pil_im)

    # Extract and save features.
    extracted_features = extractor_fn(im, resize_factor)
    if config.use_global_features:
      global_descriptor = extracted_features['global_descriptor']
      datum_io.WriteToFile(global_descriptor, output_global_feature_filename)
    if config.use_local_features:
      locations = extracted_features['local_features']['locations']
      descriptors = extracted_features['local_features']['descriptors']
      feature_scales = extracted_features['local_features']['scales']
      attention = extracted_features['local_features']['attention']
      feature_io.WriteToFile(output_local_feature_filename, locations,
                             feature_scales, descriptors, attention)
41/96:
for i in range(num_images):
    if i == 0:
      print('Starting to extract features...')
    elif i % _STATUS_CHECK_ITERATIONS == 0:
      elapsed = (time.time() - start)
      print('Processing image %d out of %d, last %d '
            'images took %f seconds' %
            (i, num_images, _STATUS_CHECK_ITERATIONS, elapsed))
      start = time.time()
    line = image_list[i]
    image_path, image_label, image_width, image_height = line.split(',')
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    input_image_filename = os.path.join(images_dir, image_path)
    if not os.path.exists(input_image_filename):
        missing_images.append(image_name)
        continue
    
    # Compose output file name and decide if image should be skipped.
    should_skip_global = True
    should_skip_local = True
    if config.use_global_features:
      output_global_feature_filename = os.path.join(output_features_dir, image_name + _DELG_GLOBAL_EXTENSION)
      if not tf.io.gfile.exists(output_global_feature_filename):
        should_skip_global = False
    if config.use_local_features:
      output_local_feature_filename = os.path.join(output_features_dir, image_name + _DELG_LOCAL_EXTENSION)
      if not tf.io.gfile.exists(output_local_feature_filename):
        should_skip_local = False
    if should_skip_global and should_skip_local:
      print('Skipping %s' % image_name)
      continue

    pil_im = utils.RgbLoader(input_image_filename)
    resize_factor = 1.0
    # if image_set == 'query':
    #   # Crop query image according to bounding box.
    #   original_image_size = max(pil_im.size)
    #   bbox = [int(round(b)) for b in ground_truth[i]['bbx']]
    #   pil_im = pil_im.crop(bbox)
    #   cropped_image_size = max(pil_im.size)
    #   resize_factor = cropped_image_size / original_image_size
    im = np.array(pil_im)

    # Extract and save features.
    extracted_features = extractor_fn(im, resize_factor)
    if config.use_global_features:
      global_descriptor = extracted_features['global_descriptor']
      datum_io.WriteToFile(global_descriptor, output_global_feature_filename)
    if config.use_local_features:
      locations = extracted_features['local_features']['locations']
      descriptors = extracted_features['local_features']['descriptors']
      feature_scales = extracted_features['local_features']['scales']
      attention = extracted_features['local_features']['attention']
      feature_io.WriteToFile(output_local_feature_filename, locations,
                             feature_scales, descriptors, attention)
41/97:
  for i in range(num_images):
    if i == 0:
      print('Starting to extract features...')
    elif i % _STATUS_CHECK_ITERATIONS == 0:
      elapsed = (time.time() - start)
      print('Processing image %d out of %d, last %d '
            'images took %f seconds' %
            (i, num_images, _STATUS_CHECK_ITERATIONS, elapsed))
      start = time.time()
    line = image_list[i]
    image_path, image_label, image_width, image_height = line.split(';;')
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    input_image_filename = os.path.join(images_dir, image_path)
    if not os.path.exists(input_image_filename):
        missing_images.append(image_name)
        continue
    
    # Compose output file name and decide if image should be skipped.
    should_skip_global = True
    should_skip_local = True
    if config.use_global_features:
      output_global_feature_filename = os.path.join(output_features_dir, image_name + _DELG_GLOBAL_EXTENSION)
      if not tf.io.gfile.exists(output_global_feature_filename):
        should_skip_global = False
    if config.use_local_features:
      output_local_feature_filename = os.path.join(output_features_dir, image_name + _DELG_LOCAL_EXTENSION)
      if not tf.io.gfile.exists(output_local_feature_filename):
        should_skip_local = False
    if should_skip_global and should_skip_local:
      print('Skipping %s' % image_name)
      continue

    pil_im = utils.RgbLoader(input_image_filename)
    resize_factor = 1.0
    # if image_set == 'query':
    #   # Crop query image according to bounding box.
    #   original_image_size = max(pil_im.size)
    #   bbox = [int(round(b)) for b in ground_truth[i]['bbx']]
    #   pil_im = pil_im.crop(bbox)
    #   cropped_image_size = max(pil_im.size)
    #   resize_factor = cropped_image_size / original_image_size
    im = np.array(pil_im)

    # Extract and save features.
    extracted_features = extractor_fn(im, resize_factor)
    if config.use_global_features:
      global_descriptor = extracted_features['global_descriptor']
      datum_io.WriteToFile(global_descriptor, output_global_feature_filename)
    if config.use_local_features:
      locations = extracted_features['local_features']['locations']
      descriptors = extracted_features['local_features']['descriptors']
      feature_scales = extracted_features['local_features']['scales']
      attention = extracted_features['local_features']['attention']
      feature_io.WriteToFile(output_local_feature_filename, locations,
                             feature_scales, descriptors, attention)
41/98:
if i == 0:
  print('Starting to extract features...')
elif i % _STATUS_CHECK_ITERATIONS == 0:
  elapsed = (time.time() - start)
  print('Processing image %d out of %d, last %d '
        'images took %f seconds' %
        (i, num_images, _STATUS_CHECK_ITERATIONS, elapsed))
  start = time.time()
line = image_list[i]
image_path, image_label, image_width, image_height = line.split(';;')
image_name = os.path.splitext(os.path.basename(image_path))[0]
input_image_filename = os.path.join(images_dir, image_path)
if not os.path.exists(input_image_filename):
    missing_images.append(image_name)
    continue

# Compose output file name and decide if image should be skipped.
should_skip_global = True
should_skip_local = True
if config.use_global_features:
  output_global_feature_filename = os.path.join(output_features_dir, image_name + _DELG_GLOBAL_EXTENSION)
  if not tf.io.gfile.exists(output_global_feature_filename):
    should_skip_global = False
if config.use_local_features:
  output_local_feature_filename = os.path.join(output_features_dir, image_name + _DELG_LOCAL_EXTENSION)
  if not tf.io.gfile.exists(output_local_feature_filename):
    should_skip_local = False
if should_skip_global and should_skip_local:
  print('Skipping %s' % image_name)
  continue

pil_im = utils.RgbLoader(input_image_filename)
resize_factor = 1.0
# if image_set == 'query':
#   # Crop query image according to bounding box.
#   original_image_size = max(pil_im.size)
#   bbox = [int(round(b)) for b in ground_truth[i]['bbx']]
#   pil_im = pil_im.crop(bbox)
#   cropped_image_size = max(pil_im.size)
#   resize_factor = cropped_image_size / original_image_size
im = np.array(pil_im)

# Extract and save features.
extracted_features = extractor_fn(im, resize_factor)
if config.use_global_features:
  global_descriptor = extracted_features['global_descriptor']
  datum_io.WriteToFile(global_descriptor, output_global_feature_filename)
if config.use_local_features:
  locations = extracted_features['local_features']['locations']
  descriptors = extracted_features['local_features']['descriptors']
  feature_scales = extracted_features['local_features']['scales']
  attention = extracted_features['local_features']['attention']
  feature_io.WriteToFile(output_local_feature_filename, locations,
                         feature_scales, descriptors, attention)
41/99: i = 0
41/100:
if i == 0:
  print('Starting to extract features...')
elif i % _STATUS_CHECK_ITERATIONS == 0:
  elapsed = (time.time() - start)
  print('Processing image %d out of %d, last %d '
        'images took %f seconds' %
        (i, num_images, _STATUS_CHECK_ITERATIONS, elapsed))
  start = time.time()
line = image_list[i]
image_path, image_label, image_width, image_height = line.split(';;')
image_name = os.path.splitext(os.path.basename(image_path))[0]
input_image_filename = os.path.join(images_dir, image_path)

# Compose output file name and decide if image should be skipped.
should_skip_global = True
should_skip_local = True
if config.use_global_features:
  output_global_feature_filename = os.path.join(output_features_dir, image_name + _DELG_GLOBAL_EXTENSION)
  if not tf.io.gfile.exists(output_global_feature_filename):
    should_skip_global = False
if config.use_local_features:
  output_local_feature_filename = os.path.join(output_features_dir, image_name + _DELG_LOCAL_EXTENSION)
  if not tf.io.gfile.exists(output_local_feature_filename):
    should_skip_local = False
if should_skip_global and should_skip_local:
  print('Skipping %s' % image_name)
  continue

pil_im = utils.RgbLoader(input_image_filename)
resize_factor = 1.0
# if image_set == 'query':
#   # Crop query image according to bounding box.
#   original_image_size = max(pil_im.size)
#   bbox = [int(round(b)) for b in ground_truth[i]['bbx']]
#   pil_im = pil_im.crop(bbox)
#   cropped_image_size = max(pil_im.size)
#   resize_factor = cropped_image_size / original_image_size
im = np.array(pil_im)

# Extract and save features.
extracted_features = extractor_fn(im, resize_factor)
if config.use_global_features:
  global_descriptor = extracted_features['global_descriptor']
  datum_io.WriteToFile(global_descriptor, output_global_feature_filename)
if config.use_local_features:
  locations = extracted_features['local_features']['locations']
  descriptors = extracted_features['local_features']['descriptors']
  feature_scales = extracted_features['local_features']['scales']
  attention = extracted_features['local_features']['attention']
  feature_io.WriteToFile(output_local_feature_filename, locations,
                         feature_scales, descriptors, attention)
41/101:
if i == 0:
  print('Starting to extract features...')
elif i % _STATUS_CHECK_ITERATIONS == 0:
  elapsed = (time.time() - start)
  print('Processing image %d out of %d, last %d '
        'images took %f seconds' %
        (i, num_images, _STATUS_CHECK_ITERATIONS, elapsed))
  start = time.time()
line = image_list[i]
image_path, image_label, image_width, image_height = line.split(';;')
image_name = os.path.splitext(os.path.basename(image_path))[0]
input_image_filename = os.path.join(images_dir, image_path)

# Compose output file name and decide if image should be skipped.
should_skip_global = True
should_skip_local = True
if config.use_global_features:
  output_global_feature_filename = os.path.join(output_features_dir, image_name + _DELG_GLOBAL_EXTENSION)
  if not tf.io.gfile.exists(output_global_feature_filename):
    should_skip_global = False
if config.use_local_features:
  output_local_feature_filename = os.path.join(output_features_dir, image_name + _DELG_LOCAL_EXTENSION)
  if not tf.io.gfile.exists(output_local_feature_filename):
    should_skip_local = False

pil_im = utils.RgbLoader(input_image_filename)
resize_factor = 1.0
# if image_set == 'query':
#   # Crop query image according to bounding box.
#   original_image_size = max(pil_im.size)
#   bbox = [int(round(b)) for b in ground_truth[i]['bbx']]
#   pil_im = pil_im.crop(bbox)
#   cropped_image_size = max(pil_im.size)
#   resize_factor = cropped_image_size / original_image_size
im = np.array(pil_im)

# Extract and save features.
extracted_features = extractor_fn(im, resize_factor)
if config.use_global_features:
  global_descriptor = extracted_features['global_descriptor']
  datum_io.WriteToFile(global_descriptor, output_global_feature_filename)
if config.use_local_features:
  locations = extracted_features['local_features']['locations']
  descriptors = extracted_features['local_features']['descriptors']
  feature_scales = extracted_features['local_features']['scales']
  attention = extracted_features['local_features']['attention']
  feature_io.WriteToFile(output_local_feature_filename, locations,
                         feature_scales, descriptors, attention)
41/102: images_dir
41/103: images_dir = "/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/viquae_images/"
41/104:
if i == 0:
  print('Starting to extract features...')
elif i % _STATUS_CHECK_ITERATIONS == 0:
  elapsed = (time.time() - start)
  print('Processing image %d out of %d, last %d '
        'images took %f seconds' %
        (i, num_images, _STATUS_CHECK_ITERATIONS, elapsed))
  start = time.time()
line = image_list[i]
image_path, image_label, image_width, image_height = line.split(';;')
image_name = os.path.splitext(os.path.basename(image_path))[0]
input_image_filename = os.path.join(images_dir, image_path)

# Compose output file name and decide if image should be skipped.
should_skip_global = True
should_skip_local = True
if config.use_global_features:
  output_global_feature_filename = os.path.join(output_features_dir, image_name + _DELG_GLOBAL_EXTENSION)
  if not tf.io.gfile.exists(output_global_feature_filename):
    should_skip_global = False
if config.use_local_features:
  output_local_feature_filename = os.path.join(output_features_dir, image_name + _DELG_LOCAL_EXTENSION)
  if not tf.io.gfile.exists(output_local_feature_filename):
    should_skip_local = False

pil_im = utils.RgbLoader(input_image_filename)
resize_factor = 1.0
# if image_set == 'query':
#   # Crop query image according to bounding box.
#   original_image_size = max(pil_im.size)
#   bbox = [int(round(b)) for b in ground_truth[i]['bbx']]
#   pil_im = pil_im.crop(bbox)
#   cropped_image_size = max(pil_im.size)
#   resize_factor = cropped_image_size / original_image_size
im = np.array(pil_im)

# Extract and save features.
extracted_features = extractor_fn(im, resize_factor)
if config.use_global_features:
  global_descriptor = extracted_features['global_descriptor']
  datum_io.WriteToFile(global_descriptor, output_global_feature_filename)
if config.use_local_features:
  locations = extracted_features['local_features']['locations']
  descriptors = extracted_features['local_features']['descriptors']
  feature_scales = extracted_features['local_features']['scales']
  attention = extracted_features['local_features']['attention']
  feature_io.WriteToFile(output_local_feature_filename, locations,
                         feature_scales, descriptors, attention)
41/105:
if i == 0:
  print('Starting to extract features...')
elif i % _STATUS_CHECK_ITERATIONS == 0:
  elapsed = (time.time() - start)
  print('Processing image %d out of %d, last %d '
        'images took %f seconds' %
        (i, num_images, _STATUS_CHECK_ITERATIONS, elapsed))
  start = time.time()
line = image_list[i]
image_path, image_label, image_width, image_height = line.split(';;')
image_name = os.path.splitext(os.path.basename(image_path))[0]
input_image_filename = os.path.join(images_dir, image_path)

# Compose output file name and decide if image should be skipped.
should_skip_global = True
should_skip_local = True
if config.use_global_features:
  output_global_feature_filename = os.path.join(output_features_dir, image_name + _DELG_GLOBAL_EXTENSION)
  if not tf.io.gfile.exists(output_global_feature_filename):
    should_skip_global = False
if config.use_local_features:
  output_local_feature_filename = os.path.join(output_features_dir, image_name + _DELG_LOCAL_EXTENSION)
  if not tf.io.gfile.exists(output_local_feature_filename):
    should_skip_local = False

pil_im = utils.RgbLoader(input_image_filename)
resize_factor = 1.0
# if image_set == 'query':
#   # Crop query image according to bounding box.
#   original_image_size = max(pil_im.size)
#   bbox = [int(round(b)) for b in ground_truth[i]['bbx']]
#   pil_im = pil_im.crop(bbox)
#   cropped_image_size = max(pil_im.size)
#   resize_factor = cropped_image_size / original_image_size
im = np.array(pil_im)

# Extract and save features.
extracted_features = extractor_fn(im, resize_factor)
if config.use_global_features:
  global_descriptor = extracted_features['global_descriptor']
  datum_io.WriteToFile(global_descriptor, output_global_feature_filename)
if config.use_local_features:
  locations = extracted_features['local_features']['locations']
  descriptors = extracted_features['local_features']['descriptors']
  feature_scales = extracted_features['local_features']['scales']
  attention = extracted_features['local_features']['attention']
  feature_io.WriteToFile(output_local_feature_filename, locations,
                         feature_scales, descriptors, attention)
44/1:

from copy import deepcopy
from functools import partial
from pprint import pprint
import os.path as osp
44/2:
import sacred
import torch
import torch.nn as nn
from sacred import SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds
from torch.backends import cudnn
44/3:
from models.ingredient import model_ingredient, get_model
from utils import pickle_load
from utils.data.dataset_ingredient import data_ingredient, get_loaders
# from utils.training import evaluate_time as evaluate
from utils.training import evaluate
44/4: ex = sacred.Experiment('RRT Evaluation', ingredients=[data_ingredient, model_ingredient], interactive=True)
44/5:
# Filter backspaces and linefeeds
SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds
44/6:
cpu = False  # Force training on CPU
cudnn_flag = 'benchmark'
temp_dir = osp.join('logs', 'temp')
resume = None
seed = 0
44/7: device = torch.device('cuda:0' if torch.cuda.is_available() and not cpu else 'cpu')
44/8:
if cudnn_flag == 'deterministic':
  setattr(cudnn, cudnn_flag, True)
44/9:
if cudnn_flag == 'deterministic':
    setattr(cudnn, cudnn_flag, True)
44/10: torch.manual_seed(seed)
44/11: from utils.data.dataset_ingredient import get_set
44/12: from utils.data.dataset_ingredient import get_sets
44/13: (train_set, query_train_set), (query_set, gallery_set) = get_sets('r50_gldv1', '/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/oxford5k','/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/oxford5k','test_query.txt', ('test_query.txt', 'test_gallery.txt'), 'gnd_roxford5k.pkl', 500)
   1:
from copy import deepcopy
from functools import partial
from pprint import pprint
import os.path as osp
   2:
import sacred
import torch
import torch.nn as nn
from sacred import SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds
from torch.backends import cudnn
   3:
from models.ingredient import model_ingredient, get_model
from utils import pickle_load
from utils.data.dataset_ingredient import data_ingredient, get_loaders
# from utils.training import evaluate_time as evaluate
from utils.training import evaluate
   4: ex = sacred.Experiment('RRT Evaluation', ingredients=[data_ingredient, model_ingredient], interactive=True)
   5:
# Filter backspaces and linefeeds
SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds
   6:
cpu = False  # Force training on CPU
cudnn_flag = 'benchmark'
temp_dir = osp.join('logs', 'temp')
resume = None
seed = 0
   7: device = torch.device('cuda:0' if torch.cuda.is_available() and not cpu else 'cpu')
   8: torch.manual_seed(seed)
   9:
def get_sets(desc_name, 
        train_data_dir, test_data_dir, 
        train_txt, test_txt, test_gnd_file, 
        max_sequence_len):
    ####################################################################################################################################
    train_lines     = read_file(osp.join(train_data_dir, train_txt))
    train_samples   = [(line.split(',')[0], int(line.split(',')[1]), int(line.split(',')[2]), int(line.split(',')[3])) for line in train_lines]
    train_set       = FeatureDataset(train_data_dir, train_samples, desc_name, max_sequence_len)
    query_train_set = FeatureDataset(train_data_dir, train_samples, desc_name, max_sequence_len)
    ####################################################################################################################################
    test_gnd_data = None if test_gnd_file is None else pickle_load(osp.join(test_data_dir, test_gnd_file))
    query_lines   = read_file(osp.join(test_data_dir, test_txt[0]))
    gallery_lines = read_file(osp.join(test_data_dir, test_txt[1]))
    query_samples   = [(line.split(',')[0], int(line.split(',')[1]), int(line.split(',')[2]), int(line.split(',')[3])) for line in query_lines]
    gallery_samples = [(line.split(',')[0], int(line.split(',')[1]), int(line.split(',')[2]), int(line.split(',')[3])) for line in gallery_lines]
    gallery_set = FeatureDataset(test_data_dir, gallery_samples, desc_name, max_sequence_len)
    query_set   = FeatureDataset(test_data_dir, query_samples,   desc_name, max_sequence_len, gnd_data=test_gnd_data)
        
    return (train_set, query_train_set), (query_set, gallery_set)
  10: (train_set, query_train_set), (query_set, gallery_set) = get_sets('r50_gldv1', '/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/viquae_images','/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/viquae_images','test_query.txt', ('test_query.txt', 'test_gallery.txt'), None'', 500)
  11: (train_set, query_train_set), (query_set, gallery_set) = get_sets('r50_gldv1', '/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/viquae_images','/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/viquae_images','test_query.txt', ('test_query.txt', 'test_gallery.txt'), None, 500)
  12:
def read_file(filename):
    with open(filename) as f:
        lines = f.read().splitlines()
    return lines
  13: (train_set, query_train_set), (query_set, gallery_set) = get_sets('r50_gldv1', '/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/viquae_images','/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/viquae_images','test_query.txt', ('test_query.txt', 'test_gallery.txt'), None, 500)
45/1:  (train_set, query_train_set), (query_set, gallery_set) = get_sets('r50_gldv1', '/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/oxford5k','/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/oxford5k','test_query.txt', ('test_query.txt', 'test_gallery.txt'), 'gnd_roxford5k.pkl', 500)
  14:
def get_sets(desc_name, 
        train_data_dir, test_data_dir, 
        train_txt, test_txt, test_gnd_file, 
        max_sequence_len):
    ####################################################################################################################################
    train_lines     = read_file(osp.join(train_data_dir, train_txt))
    train_samples   = [(line.split(',')[0], int(line.split(',')[1]), int(line.split(',')[2]), int(line.split(',')[3])) for line in train_lines]
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
  15:
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
  16: (train_set, query_train_set), (query_set, gallery_set) = get_sets('r50_gldv1', '/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/viquae_images','/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/viquae_images','test_query.txt', ('test_query.txt', 'test_gallery.txt'), None, 500)
  17: from .dataset import FeatureDataset
  18: from utilis.data.dataset import FeatureDataset
  19: from utils.data.dataset import FeatureDataset
  20: (train_set, query_train_set), (query_set, gallery_set) = get_sets('r50_gldv1', '/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/viquae_images','/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/viquae_images','test_query.txt', ('test_query.txt', 'test_gallery.txt'), None, 500)
  21: from torch.utils.data import DataLoader, RandomSampler, BatchSampler
  22: random
  23: sampler = 'random'
  24:
if sampler == 'random':
   train_sampler = BatchSampler(RandomSampler(train_set), batch_size=batch_size, drop_last=False)
  25:
batch_size      = 36
test_batch_size = 36
max_sequence_len = 500
sampler = 'random'
  26:
if sampler == 'random':
   train_sampler = BatchSampler(RandomSampler(train_set), batch_size=batch_size, drop_last=False)
  27:
num_workers = 8  # number of workers used ot load the data
pin_memory  = True  # use the pin_memory option of DataLoader 
recalls = [1, 5, 10]
  28: num_candidates = 100
  29: train_loader = DataLoader(train_set, batch_sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)
  30: query_train_loader = DataLoader(query_train_set, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory)
  31: query_loader   = DataLoader(query_set, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory)
  32: gallery_loader = DataLoader(gallery_set, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory)
  33:
class MetricLoaders(NamedTuple):
    train: DataLoader
    num_classes: int
    query: DataLoader
    query_train: DataLoader
    gallery: Optional[DataLoader] = None
  34: from typing import NamedTuple, Optional
  35:
class MetricLoaders(NamedTuple):
    train: DataLoader
    num_classes: int
    query: DataLoader
    query_train: DataLoader
    gallery: Optional[DataLoader] = None
  36: loaders, recall_ks = MetricLoaders(train=train_loader, query_train=query_train_loader, query=query_loader, gallery=gallery_loader, num_classes=len(train_set.categories)), recalls
  37: torch.manual_seed(seed)
  38:
from .matcher import MatchERT
from sacred import Ingredient
model_ingredient = Ingredient('model')
  39:
from models.matcher import MatchERT
from sacred import Ingredient
model_ingredient = Ingredient('model')
  40:
from models.matcher import MatchERT
from sacred import Ingredient
model_ingredient = Ingredient('model', interactive=True)
  41:
def get_model(num_global_features, num_local_features, seq_len, dim_K, dim_feedforward, nhead, num_encoder_layers, dropout, activation, normalize_before):
    return MatchERT(d_global=num_global_features, d_model=num_local_features, seq_len=seq_len, d_K=dim_K, nhead=nhead, num_encoder_layers=num_encoder_layers, 
            dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, normalize_before=normalize_before)
  42:
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
  43: model = get_model(num_global_features,num_local_features,seq_len,dim_K,dim_feedforward,nhead,num_encoder_layers,dropout,activation,normalize_before)
  44:
name = 'rrt'
seq_len = 1004
dim_K = 256
dim_feedforward = 1024
nhead = 4
num_encoder_layers = 6
dropout = 0.0 
activation = "relu"
normalize_before = False
  45: model = get_model(num_global_features,num_local_features,seq_len,dim_K,dim_feedforward,nhead,num_encoder_layers,dropout,activation,normalize_before)
  46:
if resume is not None:
   checkpoint = torch.load(resume, map_location=torch.device('cpu'))
   model.load_state_dict(checkpoint['state'], strict=True)
  47: model.to(device)
  48: model.eval()
  49: loaders.query.dataset.desc_name
  50: loaders.query.dataset.data_dir
  51: nn_inds_path = osp.join(loaders.query.dataset.data_dir, 'nn_inds_%s.pkl'%loaders.query.dataset.desc_name)
  52: cache_nn_inds = torch.from_numpy(pickle_load(nn_inds_path)).long()
  53:
eval_function = partial(evaluate, model=model, 
        cache_nn_inds=cache_nn_inds,
        recall=recall_ks, query_loader=loaders.query, gallery_loader=loaders.gallery)
  54: metrics = eval_function()
  55: oxford_gnd_file = "/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/oxford5k/gnd_roxford5k.pkl"
  56: oxford_gnd = pickle_load(oxford_gnd_file)
  57: oxford_gnd
  58: oxford_gnd.keys()
  59: oxford_gnd['gnd'][0]
  60: metrics = eval_function()
  61: gnd_file = "/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/viquae_images/gnd_dev_viquae.pkl"
  62: gnd = pickle_load(gnd_file)
  63: gnd['gnd']
  64: len(oxford_gnd['gnd'])
  65: len(oxford_gnd['imlist'])
  66: len(oxford_gnd['qimlist'])
  67: len(gnd['qimlist'])
  68: import numpy as np
  69: index = np.random.choice(list(range(70)), size=len(gnd['qimlist']))
  70: len(index)
  71: index[:5]
  72: gnd['gnd'] = oxford_gnd[index]
  73: gnd['gnd'] = oxford_gnd[[index]]
  74: gnd['gnd'] = oxford_gnd['gnd'][index]
  75: gnd['gnd'] = oxford_gnd['gnd'][[index]]
  76: gnd['gnd'] = oxford_gnd['gnd'][index]
  77: index
  78: gnd['gnd'] = [oxford_gnd['gnd'][i] for i in index]
  79: gnd['gnd'][0]
  80: (train_set, query_train_set), (query_set, gallery_set) = get_sets('r50_gldv1', '/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/viquae_images','/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/viquae_images','test_query.txt', ('test_query.txt', 'test_gallery.txt'), 'gnd_dev_viquae', 500)
  81: (train_set, query_train_set), (query_set, gallery_set) = get_sets('r50_gldv1', '/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/viquae_images','/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/viquae_images','test_query.txt', ('test_query.txt', 'test_gallery.txt'), 'gnd_dev_viquae.pkl', 500)
  82: from torch.utils.data import DataLoader, RandomSampler, BatchSampler
  83:
if sampler == 'random':
   train_sampler = BatchSampler(RandomSampler(train_set), batch_size=batch_size, drop_last=False)
  84: train_loader = DataLoader(train_set, batch_sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)
  85: query_train_loader = DataLoader(query_train_set, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory)
  86: gallery_loader = DataLoader(gallery_set, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory)
  87: loaders, recall_ks = MetricLoaders(train=train_loader, query_train=query_train_loader, query=query_loader, gallery=gallery_loader, num_classes=len(train_set.categories)), recalls
  88: model = get_model(num_global_features,num_local_features,seq_len,dim_K,dim_feedforward,nhead,num_encoder_layers,dropout,activation,normalize_before)
  89: model.to(device)
  90: model.eval()
  91: cache_nn_inds = torch.from_numpy(pickle_load(nn_inds_path)).long()
  92:
eval_function = partial(evaluate, model=model, 
        cache_nn_inds=cache_nn_inds,
        recall=recall_ks, query_loader=loaders.query, gallery_loader=loaders.gallery)
  93: metrics = eval_function()
  94: loaders
  95: loaders.dataset.gnd
  96: from utilis.data.utils import json_save, pickle_save
  97: from utils.data.utils import json_save, pickle_save
  98: pickle_save(gnd_file, gnd)
  99: loaders.dataset.gnd
 100: metrics = eval_function()
 101: (train_set, query_train_set), (query_set, gallery_set) = get_sets('r50_gldv1', '/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/viquae_images','/mnt/beegfs/home/smessoud/RerankingTransformer/models/research/delf/delf/python/delg/data/viquae_images','test_query.txt', ('test_query.txt', 'test_gallery.txt'), 'gnd_dev_viquae.pkl', 500)
 102:
if sampler == 'random':
   train_sampler = BatchSampler(RandomSampler(train_set), batch_size=batch_size, drop_last=False)
 103: train_loader = DataLoader(train_set, batch_sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)
 104: query_train_loader = DataLoader(query_train_set, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory)
 105: gallery_loader = DataLoader(gallery_set, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory)
 106: loaders, recall_ks = MetricLoaders(train=train_loader, query_train=query_train_loader, query=query_loader, gallery=gallery_loader, num_classes=len(train_set.categories)), recalls
 107: model = get_model(num_global_features,num_local_features,seq_len,dim_K,dim_feedforward,nhead,num_encoder_layers,dropout,activation,normalize_before)
 108: model.to(device)
 109: model.eval()
 110: cache_nn_inds = torch.from_numpy(pickle_load(nn_inds_path)).long()
 111:
eval_function = partial(evaluate, model=model, 
        cache_nn_inds=cache_nn_inds,
        recall=recall_ks, query_loader=loaders.query, gallery_loader=loaders.gallery)
 112: metrics = eval_function()
 113: gnd['gnd'][0]
 114: gnd_file
 115: len(gnd['gnd'])
 116: ? pickle_save
 117: metrics = eval_function()
 118: gnd['gnd'][0]['junk']
 119: metrics = eval_function()
48/1: from utils import pickle_load, pickle_save, json_save, ReadSolution
 120: %history -g -f /tmp/foo.py
 121: pwd
 122: %history -g -f foo.py
