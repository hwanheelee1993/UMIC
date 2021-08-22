import argparse
import numpy as np
import os
import pickle
import json
from copy import copy
from collections import defaultdict
from pytorch_pretrained_bert import BertTokenizer
from cytoolz import curry
from tqdm import tqdm
from data.data import open_lmdb

from data import (PrefetchLoader,
                  DetectFeatLmdb, TxtTokLmdb, ItmEvalDataset, itm_eval_collate)
import shutil

@curry
def bert_tokenize(tokenizer, text):
    ids = []
    for word in text.strip().split():
        ws = tokenizer.tokenize(word)
        if not ws:
            # some special char
            continue
        ids.extend(tokenizer.convert_tokens_to_ids(ws))
    return ids


bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
tokenizer = bert_tokenize(bert_tokenizer)

parser = argparse.ArgumentParser()

parser.add_argument("--input_file",
                    default='sample.json', type=str)
parser.add_argument("--img_type",
                    default='coco_val2014', type=str)
parser.add_argument("--out_dir",
                    default='txt_db/sample', type=str)
args = parser.parse_args()

def invert_dict(d):
    d_inv = defaultdict(list)
    for k, v in d.items():
        d_inv[v].append(k)
    return d_inv

with open(args.input_file, 'r') as f:
    inputs = json.load(f)

img_ids = [x['imgid'] for x in inputs]
captions = [x['caption'] for x in inputs]

out_dir = args.out_dir

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
meta = {'annotations': ['./default.jsonl'],
 'output': './default',
 'format': 'lmdb',
 'task': 'caption_evaluation',
 'bert': 'bert-base-cased',
 'UNK': 100,
 'CLS': 101,
 'SEP': 102,
 'MASK': 103,
 'v_range': [106, 28996]}


# img2txts & txt2img
ce_txt2img = {}

img_prefix = args.img_type

for i in range(len(img_ids)):
           
    ce_txt2img[str(i)] = img_prefix+'_'+img_ids[i]+'.npz'
ce_img2txts = dict(invert_dict(ce_txt2img))


# id2len
ce_id2len = {}
sent_ids = []

for i in tqdm(range(len(img_ids))):
    sent = captions[i]
    input_ids = tokenizer(sent)
    sent_ids.append(input_ids)
    ce_id2len[str(i)] = len(input_ids)

with open(f'{out_dir}/meta.json', 'w') as f:
    json.dump(meta, f)
with open(f'{out_dir}/id2len.json', 'w') as f:
    json.dump(ce_id2len, f)
with open(f'{out_dir}/txt2img.json', 'w') as f:
    json.dump(ce_txt2img, f)
with open(f'{out_dir}/img2txts.json', 'w') as f:
    json.dump(ce_img2txts, f)


open_db = curry(open_lmdb, out_dir, readonly=False)

with open_db() as db:
    for i in tqdm(range(len(img_ids))):
        id_ = str(i)
        example = {}
        sent = captions[i].lower()
        example['input_ids'] = sent_ids[i]
        example['img_fname'] = ce_txt2img[id_]
        example['target'] = 1.0
        db[id_] = example