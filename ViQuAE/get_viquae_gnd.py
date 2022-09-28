import sys
import os
from pathlib import Path 

import numpy as np
import os.path as osp
import pickle, json, random

import torch

from datasets import load_from_disk, set_caching_enabled
set_caching_enabled(False)

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

def get_gallery_imgs_for_rrt(data_set,
                             wikipedia,
                             passage2article,
                             passage_wiki_split,
                             tuto=False):
    
    question_imgs, positive_imgs, alternative_imgs, negative_imgs, selection_imgs, is_human = [], [], [], [], [], []
    
    # for every question, get the list of the top 100 search results
    iterat = 0
    for item in data_set:
        
        if iterat >= 120 and tuto:
            break
        
        # append the question image
        question_imgs.append(item['image'])
        is_human.append(item['human'])
        
        def loop_over_passages(passages):
            
            img_list = []
            
            for passage in passages: 
                # for every passage, get the list its corresponding wikipedia article id and split
                wiki_index = int(passage2article[passage])
                
                wiki_split = passage_wiki_split[passage]
                
                wiki_item = wikipedia[wiki_split][wiki_index]

                img_list.append(wiki_item['image'])
                
            return img_list
        
        
        # append the images of passages containing the original answer
        original_answer_indices = item['search_provenance_indices']
        positive_imgs.append(loop_over_passages(original_answer_indices))
        
        # append the images of passages containing an alternative answer
        alternative_answer_indices = item['search_alternative_indices']
        alternative_imgs.append(loop_over_passages(alternative_answer_indices))
        
        # append the images of irrelevant passages
        irrelevant_indices = item['search_irrelevant_indices']
        negative_imgs.append(loop_over_passages(irrelevant_indices))
        
        # append the images of passages provided by IR search
        selection_imgs.append(loop_over_passages(item['search_indices']))
        
        iterat += 1
        
    return question_imgs, positive_imgs, alternative_imgs, negative_imgs, selection_imgs, is_human
    

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


def format_selection(selection):
    return [preserve_order(sub) for sub in selection]
    


def format_gnd_for_rrt(question_imgs, positive_imgs, alternative_imgs, negative_imgs, selection_imgs, is_human):
    
    new_gnd = {}
    new_gnd['qimlist'] = question_imgs
    new_gnd['imlist']  = list(set(question_imgs + extend(positive_imgs) + extend(alternative_imgs) + extend(negative_imgs)))
    new_gnd['simlist'] = format_selection(selection_imgs)
    new_gnd['is_human'] = is_human
    
    new_gnd_gnd =  []
    
    for i in range(len(question_imgs)):
        question_gnd = {}
        question_gnd['easy'] = list(set(positive_imgs[i]))
        question_gnd['hard'] = list(set(alternative_imgs[i]))
        question_gnd['junk'] = list(set(negative_imgs[i]) - set(positive_imgs[i]))
        question_gnd['neg']  = list(set(negative_imgs[i]) - set(positive_imgs[i])  - set(alternative_imgs[i]))
        question_gnd['provenance_entity']  = len(negative_imgs[i]) == 100
        question_gnd['ir_order']  = new_gnd['simlist'][i]
        new_gnd_gnd.append(question_gnd)
    
    new_gnd['gnd'] = new_gnd_gnd
    
    return new_gnd



def selection_imgs_ranks_for_rrt(new_gnd):
    query_names     = new_gnd['qimlist']
    selection_names = new_gnd['simlist']
    
    for i in range(len(query_names)):
        query_all_names =  preserve_order(selection_names[i] + new_gnd['gnd'][i]['easy'] + new_gnd['gnd'][i]['hard'])
        img_rank_dict = {query_all_names[k]: k for k in range(len(query_all_names))}
        rank_img_dict = {k: query_all_names[k] for k in range(len(query_all_names))}
        
        new_gnd['gnd'][i]['img_rank_dict'] = img_rank_dict
        new_gnd['gnd'][i]['rank_img_dict'] = rank_img_dict
        
        def loop_over_imgs(images):
            
            img_ranks = []
            
            for img in images:
                img_ranks.append(img_rank_dict[img])
                
            return img_ranks
        
        new_gnd['gnd'][i]['r_easy'] = loop_over_imgs(new_gnd['gnd'][i]['easy'])
        new_gnd['gnd'][i]['r_hard'] = loop_over_imgs(new_gnd['gnd'][i]['hard'])
        new_gnd['gnd'][i]['r_junk'] = loop_over_imgs(new_gnd['gnd'][i]['junk'])
        new_gnd['gnd'][i]['r_neg']  = loop_over_imgs(new_gnd['gnd'][i]['neg'])
        new_gnd['gnd'][i]['r_ir_order']  = loop_over_imgs(new_gnd['gnd'][i]['ir_order'])
        
    return new_gnd

###########################################################################
## Training on ViQuAE
###########################################################################
def prepare_gnd_for_rrt_training(new_gnd, out_gnd_file=None, save=False):
    query_names       = new_gnd['qimlist']
    gallery_names     = new_gnd['imlist']
    selection_gallery = new_gnd['simlist']
        
    for i in range(len(query_names)):
        
        query_img  = query_names[i]
        anchor_idx = gallery_names.index(query_img)
        new_gnd['gnd'][i]['anchor_idx'] = anchor_idx
                
        new_gnd['gnd'][i]['g_easy'] = [gallery_names.index(g_img) for g_img in new_gnd['gnd'][i]['easy']]
        new_gnd['gnd'][i]['g_hard'] = [gallery_names.index(g_img) for g_img in new_gnd['gnd'][i]['hard']]
        new_gnd['gnd'][i]['g_junk'] = [gallery_names.index(g_img) for g_img in new_gnd['gnd'][i]['junk']]
        new_gnd['gnd'][i]['g_neg']  = [gallery_names.index(g_img) for g_img in new_gnd['gnd'][i]['neg']]
    
    if save:
        pickle_save(out_gnd_file, new_gnd)
    
    return new_gnd



def main(argv):
    dataset_path = Path(argv[0])
    kb_path = Path(argv[1])
    wiki_path = Path(argv[2])
    output_path = Path(argv[3])
    
    dataset = load_from_disk(dataset_path)
    kb = load_from_disk(kb_path)
    wiki = load_from_disk(wiki_path)
    
    train_set, dev_set, test_set = dataset['train'], dataset['validation'], dataset['test']
    humans_with_faces, humans_without_faces, non_humans = wiki['humans_with_faces'], wiki['humans_without_faces'], wiki['non_humans']
    
    n_h_article2passage = json_load(wiki_path / 'non_humans/article2passage.json')
    n_h_passage2article = get_passage2article(n_h_article2passage)
    
    h_wo_f_article2passage = json_load(wiki_path / 'humans_without_faces/article2passage.json')
    h_wo_f_passage2article = get_passage2article(h_wo_f_article2passage)
    
    h_w_f_article2passage = json_load(wiki_path / 'humans_with_faces/article2passage.json')
    h_w_f_passage2article = get_passage2article(h_w_f_article2passage)
    
    len_n_h = len(n_h_passage2article)
    len_h_w_f = len(h_w_f_passage2article)
    len_h_wo_f = len(h_wo_f_passage2article)
    
    h_w_f_passage_split  = dict(zip(h_w_f_passage2article.keys(),  ['humans_with_faces'] * len_h_w_f))
    h_wo_f_passage_split = dict(zip(h_wo_f_passage2article.keys(), ['humans_without_faces'] * len_h_wo_f))
    n_h_passage_split    = dict(zip(n_h_passage2article.keys(),    ['non_humans'] * len_n_h))
    
    
    print('is passage2article length good', len_n_h + len_h_w_f + len_h_wo_f == 11885968)
    
    passage2article = {**h_w_f_passage2article, **h_wo_f_passage2article, **n_h_passage2article}
    print('passage2article length', len(passage2article))
    
    passage_wiki_split = {**h_w_f_passage_split, **h_wo_f_passage_split, **n_h_passage_split}
    print('passage_wiki_split lenght:', len(passage_wiki_split))
    
    json_save(wiki_path / 'passage2article.json', passage2article)
    json_save(wiki_path / 'passage_wiki_split.json', passage_wiki_split)
    
    # get image names 
    train_questions, train_positives, train_alternatives, train_negatives, train_selections, train_is_human = get_gallery_imgs_for_rrt(train_set, wiki, passage2article, passage_wiki_split)

    dev_questions, dev_positives, dev_alternatives, dev_negatives, dev_selections, dev_is_human = get_gallery_imgs_for_rrt(dev_set, wiki, passage2article, passage_wiki_split)

    test_questions, test_positives, test_alternatives, test_negatives, test_selections, test_is_human = get_gallery_imgs_for_rrt(test_set, wiki, passage2article, passage_wiki_split)

    tuto_questions, tuto_positives, tuto_alternatives, tuto_negatives, tuto_selections, tuto_is_human = get_gallery_imgs_for_rrt(train_set, wiki, passage2article, passage_wiki_split, tuto=True)

    # create and format ground truth
    train_gnd = format_gnd_for_rrt(train_questions, train_positives, train_alternatives, train_negatives, train_selections, train_is_human)

    dev_gnd = format_gnd_for_rrt(dev_questions, dev_positives, dev_alternatives, dev_negatives, dev_selections, dev_is_human)

    test_gnd = format_gnd_for_rrt(test_questions, test_positives, test_alternatives, test_negatives, test_selections, test_is_human)

    tuto_gnd = format_gnd_for_rrt(tuto_questions, tuto_positives, tuto_alternatives, tuto_negatives, tuto_selections, tuto_is_human)

    # add image ranks and image gallery indices
    train_gnd = selection_imgs_ranks_for_rrt(train_gnd)
    train_gnd = prepare_gnd_for_rrt_training(train_gnd)
    
    dev_gnd = selection_imgs_ranks_for_rrt(dev_gnd)
    dev_gnd = prepare_gnd_for_rrt_training(dev_gnd)
    
    test_gnd = selection_imgs_ranks_for_rrt(test_gnd)
    test_gnd = prepare_gnd_for_rrt_training(test_gnd)
    
    tuto_gnd = selection_imgs_ranks_for_rrt(tuto_gnd)
    tuto_gnd = prepare_gnd_for_rrt_training(tuto_gnd)
    
    # saving files
    train_gnd_file = output_path / "gnd_train.pkl"
    pickle_save(train_gnd_file, train_gnd)

    dev_gnd_file = output_path / "gnd_dev.pkl"
    pickle_save(dev_gnd_file, dev_gnd)

    test_gnd_file = output_path / "gnd_test.pkl"
    pickle_save(test_gnd_file, test_gnd)

    tuto_gnd_file = output_path / "gnd_tuto.pkl"
    pickle_save(tuto_gnd_file, tuto_gnd)
    
    entire_dataset_imgs = list(set(train_gnd['imlist'] + dev_gnd['imlist'] + test_gnd['imlist']))
    np.savetxt(output_path / 'images.txt', entire_dataset_imgs, fmt="%s")


    
if __name__ == "__main__":
    main(sys.argv[1:])