"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

CE dataset
"""
from collections import defaultdict
import copy
import random
import json
import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from cytoolz import concat
import numpy as np
import pickle
from .data import (DetectFeatTxtTokDataset, DetectFeatLmdb, TxtTokLmdb,
                   pad_tensors, get_gather_index, get_ids_and_lens)
from .sampler import TokenBucketSampler
#from transformers import AutoTokenizer
from pytorch_pretrained_bert import BertTokenizer
import nltk
#tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

cutoff = ['CC', 'VBG', 'IN', 'RP']

#with open('/txt/tag_word_counter.pkl', 'rb') as f:
#    tag_word_counter = pickle.load(f)

def postags(text):
    sentences = nltk.word_tokenize(text)
    sentences = nltk.pos_tag(sentences)
    return sentences

def distract(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)

    err = 0
    tags = [x[1] for x in sent]
    indexes =[i for i in range(len(tags)) if tags[i] in cutoff]
    if len(indexes) < 1:
        err = 1
        outputs = [x[0] for x in sent][:2*int(len(sent)/3)]
    else:
        outputs = [x[0] for x in sent][:np.random.choice(indexes)]
    return ' '.join(outputs)

def make_st(org_sent, change_prob=70):
    tokenized = nltk.word_tokenize(org_sent)
    postags = nltk.pos_tag(tokenized)
    tokens = [x[0] for x in postags]
    tags = [x[1] for x in postags]
    changed = False
    for i, x in enumerate(tags):
        if x.startswith('V') or x.startswith('N') or x.startswith('J'):
            if(np.random.randint(100) > change_prob):
                neg_word = random.choices(population=list(tag_word_counter[x].keys()), weights=list(tag_word_counter[x].values()))[0]
                if(neg_word != x):
                    changed = True
                tokens[i] = neg_word
    return (' '.join(tokens)), changed

def make_rd(org_sent):
    tokenized = nltk.word_tokenize(org_sent)
    postags = nltk.pos_tag(tokenized)
    tokens = [x[0] for x in postags]
    tags = [x[1] for x in postags]
    changed = False
    for i in range(len(tags)):
        if tags[i].startswith('V') or tags[i].startswith('N') or tags[i].startswith('J'):
            if(np.random.uniform()> 0.7):
                changed = True
                if(np.random.uniform()>0.5):
                    tokens[i] = tokens[i]+' '+tokens[i]
                else:
                    tokens[i] = ''
    return (' '.join(tokens)).replace('  ',' '), changed


def non_fluent(sent, pd_prob=0.5, leave_prob=0.7):
    if(np.random.uniform() > pd_prob):
        sent_tokens = sent.split(' ')
        random.shuffle(sent_tokens)
        sent = ' '.join(sent_tokens)    

    else:
        changed = False
        while(True):
            sent_tokens = sent.split(' ')
            del_prob = np.random.binomial(1, leave_prob, len(sent_tokens))
            dist_tokens = []
            for i in range(len(sent_tokens)):
                if(del_prob[i]):
                    dist_tokens.append(sent_tokens[i])
            if(dist_tokens != sent_tokens):
                break
                
        sent = ' '.join(dist_tokens)
    return sent

def _has_overlap(la, lb):
    if len(la) < len(lb):
        la, lb = lb, la
    s = set(la)
    return any(b in s for b in lb)

def sample_negative(sample_pool, ground_truths, num_sample):
    """ random and retry """
    outputs = ground_truths[:1]
    while _has_overlap(outputs, ground_truths):
        outputs = random.sample(sample_pool, num_sample)
    return outputs

def _get_ce_target(example):
    target = torch.zeros(2)
    labels = torch.tensor([float(example['target'])])
    scores = None
    if labels and scores:
        target.scatter_(0, torch.tensor(labels), torch.tensor(scores))
        return target
    else:
        return labels


class CeDataset(DetectFeatTxtTokDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, i):
        example = super().__getitem__(i)
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'])

        # text input
        input_ids = example['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)

        target = _get_ce_target(example)

        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        return input_ids, img_feat, img_pos_feat, attn_masks, target


def ce_collate(inputs):
    (input_ids, img_feats, img_pos_feats, attn_masks, targets
     ) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    targets = torch.stack(targets, dim=0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'targets': targets}
    return batch


class CeRankDataset(DetectFeatTxtTokDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, i):
        example = super().__getitem__(i)
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'])

        # text input
        input_ids = example['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)

        target = _get_ce_target(example)

        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        return input_ids, img_feat, img_pos_feat, attn_masks, target


def ce_rank_collate(inputs):
    (input_ids, img_feats, img_pos_feats, attn_masks, targets
     ) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    targets = torch.stack(targets, dim=0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'targets': targets,
             'sample_size' : 5}
    return batch
   
class CeRankItmDataset(DetectFeatTxtTokDataset):
    def __init__(self, txt_db, img_db, neg_sample_size=1):
        super().__init__(txt_db, img_db)

        txt2img = self.txt_db.txt2img
        self.txt2img = {id_: txt2img[id_] for id_ in self.ids}
        # images partitioned by rank
        self.img2txts = defaultdict(list)
        for id_, img in self.txt2img.items():
            self.img2txts[img].append(id_)
        self.img_name_list = list(self.img2txts.keys())

        self.neg_sample_size = neg_sample_size
       
    def __getitem__(self, i):
        gt_txt_id = self.ids[i]
        gt_img_fname = self.txt2img[gt_txt_id]

        id_pairs = [(gt_txt_id, gt_img_fname)]
        # sample negatives
        neg_sample_txt_ids = sample_negative(
            self.ids, self.img2txts[gt_img_fname], self.neg_sample_size)
        id_pairs.extend([(neg_txt_id, gt_img_fname)
                         for neg_txt_id in neg_sample_txt_ids])
        inputs = self._collect_inputs(id_pairs)
        assert len(inputs) == (1 + self.neg_sample_size)
        return inputs

    def _collect_inputs(self, id_pairs):
        # create input features
        inputs = []
        for txt_id, img_id in id_pairs:
            example = self.txt_db[txt_id]
            # text input
            input_ids = example['input_ids']
            input_ids = self.txt_db.combine_inputs(input_ids)
            # img input
            img_feat, img_pos_feat, num_bb = self._get_img_feat(img_id)
            # mask
            attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

            inputs.append((input_ids, img_feat, img_pos_feat, attn_masks, txt_id))

        return inputs


def ce_rank_itm_collate(inputs):
    (input_ids, img_feats, img_pos_feats, attn_masks, txt_id
     ) = map(list, unzip(concat(i for i in inputs)))

    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    sample_size = len(inputs[0])
    assert all(sample_size == len(i) for i in inputs)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'sample_size': 10}
    return batch


class CeEvalDataset(CeDataset):
    def __init__(self, *args, is_ce=False, **kwargs):
        self.is_ce = is_ce
        super().__init__(*args, **kwargs)
        
    def __getitem__(self, i):
        qid = self.ids[i]
        example = DetectFeatTxtTokDataset.__getitem__(self, i)
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'])

        # text input
        input_ids = example['input_ids']
        #input_ids = input_ids + (60-len(input_ids))*[0]
        input_ids = self.txt_db.combine_inputs(input_ids)

        if 'target' in example:
            if(self.is_ce):
                target = torch.tensor([example['target']])
            else:
                target = _get_ce_target(example)
        else:
            target = None

        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        return qid, input_ids, img_feat, img_pos_feat, attn_masks, target


def ce_eval_collate(inputs):
    (qids, input_ids, img_feats, img_pos_feats, attn_masks, targets
     ) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)
    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    if targets[0] is None:
        targets = None
    else:
        targets = torch.stack(targets, dim=0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'qids': qids,
             'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'targets': targets,
             'sample_size' : 5}
    return batch


class CeRankNegDataset(DetectFeatTxtTokDataset):
    def __init__(self, txt_db, img_db, split='train', neg_sample_size=1):
        super().__init__(txt_db, img_db)

        split = 'all'
        with open('/storage/similar_set_id/method_retrieval_05/similar_'+split+'.json', 'r') as f:
            self.sim_cider = json.load(f)
        with open('/storage/similar_set_id/method_CIDEr_score/similar_'+split+'.json', 'r') as f:
            self.sim_ret = json.load(f)

        txt2img = self.txt_db.txt2img
        self.txt2img = {id_: txt2img[id_] for id_ in self.ids}
        # images partitioned by rank
        self.img2txts = defaultdict(list)
        for id_, img in self.txt2img.items():
            self.img2txts[img].append(id_)
        self.img_name_list = list(self.img2txts.keys())

        self.neg_sample_size = neg_sample_size
       
    def __getitem__(self, i):
        gt_txt_id = self.ids[i]
        gt_img_fname = self.txt2img[gt_txt_id]

        id_pairs = [(gt_txt_id, gt_img_fname)]
        # sample negatives
        neg_sample_txt_ids = sample_negative(
            self.ids, self.img2txts[gt_img_fname], self.neg_sample_size)
        id_pairs.extend([(neg_txt_id, gt_img_fname)
                         for neg_txt_id in neg_sample_txt_ids])
        inputs = self._collect_inputs(id_pairs)
        #print("## len input", len(inputs))

        #assert len(inputs) == (1 + self.neg_sample_size)
        return inputs

    def _collect_inputs(self, id_pairs):
        # create input features
        

        inputs = []

        o_txt_id, o_img_id = id_pairs[0]
        o_img_id_ = o_img_id.split('_')[-1][:-4].lstrip("0")
        img_feat, img_pos_feat, num_bb = self._get_img_feat(o_img_id)

        org_input_ids = self.txt_db[o_txt_id]['input_ids']
        org_cap = tokenizer.decode(org_input_ids, skip_special_tokens=True)

        is_print=False

        if(is_print):
            print("0. Org cap: ",org_cap)

        ct = 0
        # Original + Random Negative
        for txt_id, img_id in id_pairs:
            ct += 1
            example = self.txt_db[txt_id]
            # text input
            input_ids = example['input_ids']
            if(is_print and ct == 2):
                rand_cap = tokenizer.decode(input_ids, skip_special_tokens=True)
                print("1. Rand cap: ",rand_cap)
            input_ids = self.txt_db.combine_inputs(input_ids)
            # img input
            # mask
            attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

            inputs.append((input_ids, img_feat, img_pos_feat, attn_masks, txt_id))

        n_hard_neg = 3
        ## - Using Hard Negative
        if(np.random.randint(100) % 2 == 0):
            #top_img_id = self.sim_ret[o_img_id_][0]
            top_img_ids = self.sim_ret[o_img_id_][:n_hard_neg]
            neg_type = 'R'
        else:
            #top_img_id = self.sim_cider[o_img_id_][0]
            top_img_ids = self.sim_cider[o_img_id_][:n_hard_neg]
            neg_type = 'C'

        prefix = o_img_id[:o_img_id.index('4')+2]
        #top_img_f = prefix + str(top_img_id).zfill(12)+'.npz'

        sample_from_hard_neg = False
        top_img_fs = []
        for top_img_id in top_img_ids:
            top_img_f = prefix + str(top_img_id).zfill(12)+'.npz'
            if(top_img_f in list(self.img2txts.keys())):
                sample_from_hard_neg = True
                top_img_fs.append(top_img_f)

        if(sample_from_hard_neg):
            sampled_img_f = np.random.choice(top_img_fs)
            neg_id = np.random.choice(self.img2txts[sampled_img_f])
        else:
            neg_id = np.random.choice(self.ids)
        neg_example = self.txt_db[neg_id]
        # text input
        input_ids = neg_example['input_ids']
        neg_cap = tokenizer.decode(input_ids, skip_special_tokens=True)

        if(is_print):
            print("2. Hard Neg", neg_type, ': ',neg_cap)

        input_ids = self.txt_db.combine_inputs(input_ids)
        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)
        inputs.append((input_ids, img_feat, img_pos_feat, attn_masks, o_txt_id))   

        ## - Using Subtitution based POS-TAGS
        while(True):
            neg_cap, changed = make_st(org_cap)
            if(changed):
                break
        if(is_print):
            print("3. Substitution: ", neg_cap, '\n')
        input_ids = tokenizer.encode(neg_cap)
        #input_ids = input_ids + (60-len(input_ids))*[0]
        input_ids = self.txt_db.combine_inputs(input_ids)
        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)
        inputs.append((input_ids, img_feat, img_pos_feat, attn_masks, o_txt_id))

        ## - Using Non-Fluent Captions
        neg_cap = non_fluent(org_cap)
        if(is_print):
            print("4. Non-Fluent: ", neg_cap, '\n')
        input_ids = tokenizer.encode(neg_cap)
        input_ids = self.txt_db.combine_inputs(input_ids)
        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)
        inputs.append((input_ids, img_feat, img_pos_feat, attn_masks, o_txt_id))
        
        ## - Using Repeat or Negation
        while(True):
            neg_cap, changed = make_rd(org_cap)
            if(changed):
                break
        if(is_print):
            print("5. Repeat and Delete: ", neg_cap, '\n')
        input_ids = tokenizer.encode(neg_cap)
        #input_ids = input_ids + (60-len(input_ids))*[0]
        input_ids = self.txt_db.combine_inputs(input_ids)
        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)
        inputs.append((input_ids, img_feat, img_pos_feat, attn_masks, o_txt_id))
      
        return inputs


def ce_rank_neg_collate(inputs):
    (input_ids, img_feats, img_pos_feats, attn_masks, txt_id
     ) = map(list, unzip(concat(i for i in inputs)))

    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    sample_size = len(inputs[0])
    assert all(sample_size == len(i) for i in inputs)
    
    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'sample_size': sample_size}
    return batch
