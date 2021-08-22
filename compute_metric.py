import argparse
import json
import os
from os.path import exists
import pickle
from time import time
import math
import torch
from torch.utils.data import DataLoader

from horovod import torch as hvd

from data import (PrefetchLoader,
                  DetectFeatLmdb, TxtTokLmdb, ItmEvalDataset, itm_eval_collate,
                 CeEvalDataset, ce_eval_collate)
from model.ce import UniterForCaptioningMetric
from utils.distributed import all_gather_list
from utils.const import IMG_DIM
from utils.itm_eval import inference, itm_eval
from types import SimpleNamespace
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.nn.functional import softmax
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as scss
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt",
                    default='ckpt/umic.pt', type=str)
parser.add_argument("--img_db",
                    default='img_db/coco_val2014', type=str)
parser.add_argument("--txt_db",
                    default='txt_db/capeval1k', type=str)
parser.add_argument("--batch_size",
                    default=128, type=int)
parser.add_argument("--out_file",
                    default='umic_capeval1k.json', type=str)


args = parser.parse_args()

def sigmoid(x):
    return 1/(1+math.exp(-x))

img_db_dir= args.img_db
txt_db_dir = args.txt_db

batch_size = args.batch_size
checkpoint = args.ckpt

opts = SimpleNamespace(compressed_db=False, max_txt_len=60,conf_th=0.2, max_bb=100, min_bb=10, num_bb=36, inf_minibatch_size=400, margin=0.2,
                      valid_steps=1000, n_workers=4, fp16=True,
                      img_db=img_db_dir,
                      txt_db=txt_db_dir,
                      model_config='./config/uniter-base.json',
                      output_dir='results',
                      pin_mem=True,
                      batch_size=batch_size,
                      checkpoint=checkpoint)
hvd.init()
n_gpu = hvd.size()
device = torch.device("cuda", hvd.local_rank())
torch.cuda.set_device(hvd.local_rank())
rank = hvd.rank()

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

model = model.cuda()
model.eval()

eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size,
                         num_workers=1,
                         pin_memory=False,
                         collate_fn=ce_eval_collate)
eval_dataloader = PrefetchLoader(eval_dataloader)

umic_scores = []
for i, batch, in tqdm(enumerate(eval_dataloader)):
    with torch.no_grad():
        scores = model(batch, compute_loss=False)
        umic_scores += (list(scores.squeeze().detach().cpu().numpy()))
        
umic_scores = [sigmoid(x) for x in umic_scores]

print("UMIC Score: %.3f"% np.average(umic_scores))

# Save the scores
with open(args.out_file, 'w') as f:
    json.dump(umic_scores, f)
