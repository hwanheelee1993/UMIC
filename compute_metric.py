import argparse
import json
import os
from os.path import exists
import pickle
from time import time
import math
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax

from apex import amp
from horovod import torch as hvd

from data import (PrefetchLoader,
                  DetectFeatLmdb, TxtTokLmdb, ItmEvalDataset, itm_eval_collate,
                 CeEvalDataset, ce_eval_collate)
from model.ce import UniterForCaptioningMetric
from utils.logger import LOGGER
from utils.distributed import all_gather_list
from utils.misc import Struct
from utils.const import IMG_DIM
from utils.itm_eval import inference, itm_eval
from types import SimpleNamespace
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as scss
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument("--data_type",
                    default='capeval1k', type=str)
parser.add_argument("--ckpt",
                    default='/storage/umic.pt', type=str)
parser.add_argument("--img_type",
                    default='coco_val2014', type=str)

args = parser.parse_args()

def sigmoid(x):
    return 1/(1+math.exp(-x))

data_type = args.data_type
img_prefix = args.img_type

img_db_dir='/img/'+img_prefix+'/'
txt_db_dir ='/txt/ce_'+data_type+'.db'
batch_size = 100

hevalf = '/captions/'+args.data_type+'.pkl'

with open(hevalf, 'rb') as f:
    out_dict = pickle.load(f)
    
checkpoint = args.ckpt

opts = SimpleNamespace(compressed_db=False, max_txt_len=60,conf_th=0.2, max_bb=100, min_bb=10, num_bb=36, inf_minibatch_size=400, margin=0.2,
                      valid_steps=1000, n_workers=4, fp16=True,
                      img_db=img_db_dir,
                      txt_db=txt_db_dir,
                      model_config='/src/config/uniter-base.json',
                      output_dir='results',
                      pin_mem=True,
                      batch_size=batch_size,
                      checkpoint=checkpoint)
hvd.init()
n_gpu = hvd.size()
device = torch.device("cuda", hvd.local_rank())
torch.cuda.set_device(hvd.local_rank())
rank = hvd.rank()
LOGGER.info("device: {} n_gpu: {}, rank: {}, "
            "16-bits training: {}".format(
                device, n_gpu, hvd.rank(), opts.fp16)) 

# load DBs and image dirs
eval_img_db = DetectFeatLmdb(opts.img_db,
                             opts.conf_th, opts.max_bb,
                             opts.min_bb, opts.num_bb,
                             opts.compressed_db)
eval_txt_db = TxtTokLmdb(opts.txt_db, -1)
eval_dataset = CeEvalDataset(eval_txt_db, eval_img_db)

# Prepare model
load_checkpoint = torch.load(opts.checkpoint)

model = UniterForCaptioningMetric.from_pretrained(
    opts.model_config, load_checkpoint, img_dim=IMG_DIM)

model.to(device)
model = amp.initialize(model, enabled=opts.fp16, opt_level='O2')
model.eval()

eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size,
                         num_workers=1,
                         pin_memory=False,
                         collate_fn=ce_eval_collate)
eval_dataloader = PrefetchLoader(eval_dataloader)

p_scores = []
h_scores = []
for i, batch, in tqdm(enumerate(eval_dataloader)):
    with torch.no_grad():
        scores = model(batch, compute_loss=False)
        h_scores += (list(batch['targets'][:,0].detach().cpu().numpy()))
        p_scores += (list(scores.squeeze().detach().cpu().numpy()))
        

oscores = p_scores
sgscores = [sigmoid(x) for x in oscores]

# Save the scores
with open('./scores/'+args.data_type+'.pkl', 'wb') as f:
    pickle.dump(sgscores, f)