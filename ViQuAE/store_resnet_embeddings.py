import sys
import os
from pathlib import Path 

import numpy as np
from tqdm import tqdm
import os.path as osp
import pickle, json, random

import torch
from pprint import pprint
from utils.data.delf import datum_io
from copy import deepcopy

from datasets import load_from_disk, set_caching_enabled
set_caching_enabled(False)
_IMAGENET_EXTENSION = '.imagenet'

###############################################################
def pickle_load(path):
    with open(path, 'rb') as fid:
        data_ = pickle.load(fid)
    return data_


def pickle_save(path, data):
    with open(path, 'wb') as fid:
        pickle.dump(data, fid)


def json_load(path):
    with open(path, 'r') as fid:
        data_ = json.load(fid)
    return data_


def json_save(path, data):
    with open(path, 'w') as fid:
        json.dump(data, fid, indent=4, sort_keys=True)
###############################################################

def get_gallery_img_embeddings(data_set,
                                wikipedia,
                                passage2article,
                                passage_wiki_split,
                                tuto=False,
                                img_embeddings = {}):
    
    # for every question, get the list of the top 100 search results
    iterat = 0
    for item in data_set:
        
        if iterat >= 120 and tuto:
            break
        
        # append the question image
        img_embeddings[item['image']] = item['keep_image_embedding']
        
        def loop_over_passages(passages, img_embeddings):
            
            for passage in passages: 
                # for every passage, get the list its corresponding wikipedia article id and split
                wiki_index = passage2article[str(passage)]
                
                wiki_split = passage_wiki_split[str(passage)]
                
                wiki_item = wikipedia[wiki_split][int(wiki_index)]

                img_embeddings[wiki_item['image']] = wiki_item['image_embedding']
                
            return img_embeddings
        
        
        # append the images of passages containing the original answer
        original_answer_indices = item['search_provenance_indices']
        img_embeddings = loop_over_passages(original_answer_indices, img_embeddings)
        
        # append the images of passages containing an alternative answer
        alternative_answer_indices = item['search_alternative_indices']
        img_embeddings = loop_over_passages(alternative_answer_indices, img_embeddings)
        
        # append the images of irrelevant passages
        irrelevant_indices = item['search_irrelevant_indices']
        img_embeddings = loop_over_passages(irrelevant_indices, img_embeddings)
        
        # append the images of passages provided by IR search
        img_embeddings = loop_over_passages(item['search_indices'], img_embeddings)
        
        iterat += 1
        
    return img_embeddings
     

def get_passage2article(article2passage):
    passage2article = {} 
    for k, v in article2passage.items(): 
        for x in v: 
            passage2article[x] = k

    return passage2article

def extend(a):
    out = []
    for sublist in a:
        out.extend(sublist)
    return out

def preserve_order(array):
    new_array = []
    for e in array:
        if e in new_array:
            continue
        else:
            new_array.append(e)
    return new_array



def main(argv):
    dataset_path = Path(argv[0])
    kb_path = Path(argv[1])
    wiki_path = Path(argv[2])
    output_path = Path(argv[3])
    
    dataset = load_from_disk(dataset_path)
    kb = load_from_disk(kb_path)
    wiki = load_from_disk(wiki_path)
    
    train_set, dev_set, test_set = dataset['train'], dataset['validation'], dataset['test']
    
    passage2article = json_load(wiki_path / 'passage2article.json')
    print('passage2article length', len(passage2article))
    passage_wiki_split = json_load(wiki_path / 'passage_wiki_split.json')
    print('passage_wiki_split length', len(passage_wiki_split))
    
    train_img_embeddings = get_gallery_img_embeddings(train_set, wiki, passage2article, passage_wiki_split)
    dev_img_embeddings   = get_gallery_img_embeddings(dev_set, wiki,   passage2article, passage_wiki_split)
    test_img_embeddings  = get_gallery_img_embeddings(test_set, wiki,  passage2article, passage_wiki_split)
    
    dataset_img_embeddings = {**train_img_embeddings, **dev_img_embeddings, **test_img_embeddings}
    
    for image_name, embedding in dataset_img_embeddings.items():
        image_name = '.'.join(image_name.split('.')[:-1])
        output_feature_filename = output_path / (image_name + _IMAGENET_EXTENSION)
        datum_io.WriteToFile(np.array(embedding), str(output_feature_filename))
    
    
if __name__ == "__main__":
    main(sys.argv[1:])