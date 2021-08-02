"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

UMIC
"""
import argparse
import json
import os
from os.path import exists, join
import pickle
from time import time
import numpy as np

import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, ConcatDataset

from apex import amp
from horovod import torch as hvd
import pandas as pd
from tqdm import tqdm

from data import (TokenBucketSampler, PrefetchLoader, ImageLmdbGroup,
                  DetectFeatLmdb, TxtTokLmdb,
                  CeRankItmDataset, CeRankNegDataset,
                  ce_rank_itm_collate, ce_rank_neg_collate,
                  CeEvalDataset, ce_eval_collate)
from model.ce import UniterForCaptioningMetric
from optim import get_lr_sched
from optim.misc import build_optimizer

from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.distributed import (all_reduce_and_rescale_tensors, all_gather_list,
                               broadcast_tensors)
from utils.save import ModelSaver, save_training_meta
from utils.misc import NoOp, parse_with_config, set_dropout, set_random_seed
from utils.const import IMG_DIM, BUCKET_SIZE
import numpy as np
label2ans = {
    0:0,
    1:1
}


def create_dataloader(img_paths, txt_paths, batch_size, is_train,
                      dset_cls, collate_fn, opts):
    #####

    all_img_dbs = ImageLmdbGroup(opts.conf_th, opts.max_bb, opts.min_bb,
                                 opts.num_bb, opts.compressed_db)
    datasets = []
    for txt_path_, img_path_ in zip(txt_paths, img_paths):
        img_db = all_img_dbs[img_path_]
        txt_db = TxtTokLmdb(txt_path_, opts.max_txt_len)
        datasets.append(dset_cls(txt_db, img_db))

    dset = ConcatDataset(datasets)   
    loader = DataLoader(dset, batch_size=batch_size, shuffle=False, drop_last=is_train,
                        num_workers=opts.n_workers, pin_memory=opts.pin_mem,
                        collate_fn=collate_fn)
    return PrefetchLoader(loader)


def main(opts):
    hvd.init()
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    rank = hvd.rank()
    opts.rank = rank
    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, hvd.rank(), opts.fp16))

    if opts.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, "
                         "should be >= 1".format(
                            opts.gradient_accumulation_steps))

    set_random_seed(opts.seed)

    # train_examples = None
    LOGGER.info(f"Loading Train Dataset {opts.train_txt_db}, "
                f"{opts.train_img_db}")
    train_dataloader = create_dataloader(opts.train_img_db, opts.train_txt_db,
                                         opts.train_batch_size, True,
                                         CeRankNegDataset, ce_rank_neg_collate, opts)
    val_dataloader = create_dataloader(opts.val_img_db, opts.val_txt_db,
                                       opts.val_batch_size, False,
                                       CeRankNegDataset, ce_rank_neg_collate, opts)
    test_dataloader = create_dataloader(opts.test_img_db, opts.test_txt_db,
                                        opts.val_batch_size, False,
                                        CeEvalDataset, ce_eval_collate, opts)

    # Prepare model
    if opts.checkpoint:
        checkpoint = torch.load(opts.checkpoint)
    else:
        checkpoint = {}
    #bert_model = json.load(open(f'{opts.train_txt_db}/meta.json'))['bert']
    #if 'bert' not in bert_model:
    bert_model = 'bert-base-cased'  # quick hack for glove exp
    
    model = UniterForCaptioningMetric.from_pretrained(
        opts.model_config, state_dict=checkpoint,
        img_dim=IMG_DIM, margin=opts.margin)
    #if not opts.checkpoint:
    model.init_output() # for initialization of rank layer
    model.to(device)
    # make sure every process has same model parameters in the beginning
    broadcast_tensors([p.data for p in model.parameters()], 0)
    set_dropout(model, opts.dropout)

    # Prepare optimizer
    optimizer = build_optimizer(model, opts)
    model, optimizer = amp.initialize(model, optimizer,
                                      enabled=opts.fp16, opt_level='O2')

    global_step = 0
    if rank == 0:
        save_training_meta(opts)
        TB_LOGGER.create(join(opts.output_dir, 'log'))
        pbar = tqdm(total=opts.num_train_steps)
        model_saver = ModelSaver(join(opts.output_dir, 'ckpt'))
        
        if not os.path.exists(join(opts.output_dir, 'results')):
            os.makedirs(join(opts.output_dir, 'results'))  # store VQA predictions
        add_log_to_file(join(opts.output_dir, 'log', 'log.txt'))
    else:
        LOGGER.disabled = True
        pbar = NoOp()
        model_saver = NoOp()

    LOGGER.info(f"***** Running training with {n_gpu} GPUs *****")
    LOGGER.info("  Num examples = %d", len(train_dataloader.dataset))
    LOGGER.info("  Batch size = %d", opts.train_batch_size)
    LOGGER.info("  Accumulate steps = %d", opts.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", opts.num_train_steps)

    running_loss = RunningMeter('loss')
    n_examples = 0
    n_epoch = 0
    start = time()
    # quick hack for amp delay_unscale bug

    while True:
        model.eval()
        print("Initial Validation...")
        for split, loader in [("val", val_dataloader),
                                ("test", test_dataloader)]:
            LOGGER.info(f"Step {global_step}: start running "
                        f"validation on {split} split...")
            if(split == 'val'):
                val_log, results = validate_corr(
                    model, loader, split)
            else:
                val_log, results = validate_corr(
                    model, loader, split)

            TB_LOGGER.log_scaler_dict(val_log)
        model.train()

        optimizer.zero_grad()

        for step, batch in enumerate(train_dataloader):
            n_examples += batch['input_ids'].size(0)
            loss, _ = model(batch, compute_loss=True)
            loss = loss.mean()
            delay_unscale = (step+1) % opts.gradient_accumulation_steps != 0
            with amp.scale_loss(loss, optimizer, delay_unscale=delay_unscale
                                ) as scaled_loss:
                scaled_loss.backward()
                if not delay_unscale:
                    # gather gradients from every processes
                    # do this before unscaling to make sure every process uses
                    # the same gradient scale
                    grads = [p.grad.data for p in model.parameters()
                             if p.requires_grad and p.grad is not None]
                    all_reduce_and_rescale_tensors(grads, float(1))

            running_loss(loss.item())
            if (step + 1) % opts.gradient_accumulation_steps == 0:
                global_step += 1

                # learning rate scheduling
                lr_this_step = get_lr_sched(global_step, opts)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                TB_LOGGER.add_scalar('lr', lr_this_step, global_step)

                # log loss
                # NOTE: not gathered across GPUs for efficiency
                TB_LOGGER.add_scalar('loss', running_loss.val, global_step)
                TB_LOGGER.step()

                # update model params
                if opts.grad_norm != -1:
                    grad_norm = clip_grad_norm_(amp.master_params(optimizer),
                                                opts.grad_norm)
                    TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
                optimizer.step()
                optimizer.zero_grad()
                pbar.update(1)
                del loss

                if global_step % 100 == 0:
                    # monitor training throughput
                    LOGGER.info(f'============Step {global_step}=============')
                    tot_ex = sum(all_gather_list(n_examples))
                    ex_per_sec = int(tot_ex / (time()-start))
                    LOGGER.info(f'{tot_ex} examples trained at '
                                f'{ex_per_sec} ex/s')
                    TB_LOGGER.add_scalar('perf/ex_per_s',
                                         ex_per_sec, global_step)
                    LOGGER.info(f'===========================================')

                if global_step % opts.valid_steps == 0:
                    model.eval()
                    for split, loader in [("val", val_dataloader),
                                          ("test", test_dataloader)]:
                        LOGGER.info(f"Step {global_step}: start running "
                                    f"validation on {split} split...")
                        if(split == 'val'):
                            val_log, results = validate_corr(
                                model, loader, split)
                        else:
                            val_log, results = validate_corr(
                                model, loader, split)

                        TB_LOGGER.log_scaler_dict(val_log)
                    if(global_step > 1800):
                        model_saver.save(model, global_step)
                    model.train()
            if global_step >= opts.num_train_steps:
                break
        if global_step >= opts.num_train_steps:
            break
        n_epoch += 1
        LOGGER.info(f"Step {global_step}: finished {n_epoch} epochs")
    if opts.num_train_steps % opts.valid_steps != 0:
        model.eval()
        for split, loader in [("val", val_dataloader),
                              ("test", test_dataloader)]:
            LOGGER.info(f"Step {global_step}: start running "
                        f"validation on {split} split...")
            val_log, results = validate(model, loader, split)
            TB_LOGGER.log_scaler_dict(val_log)
        model_saver.save(model, global_step)



@torch.no_grad()

def validate_corr(model, val_loader, split='val', is_print=True):
    p_scores = []
    h_scores = []

    for i, batch, in tqdm(enumerate(val_loader)):
        with torch.no_grad():
            scores = model(batch, compute_loss=False)

            h_scores += (list(batch['targets'][:,0].detach().cpu().numpy()))
            p_scores += (list(scores.squeeze().detach().cpu().numpy()))   

    df = pd.DataFrame(np.array([h_scores, p_scores]).T)

    pearson = round(df.corr('pearson')[0][1], 3)
    kendall = round(df.corr('kendall')[0][1], 3)
    spearman = round(df.corr('spearman')[0][1], 3)

    val_log = {f'valid/{split}_pearson': pearson,
               f'valid/{split}_kendall': kendall,
               f'valid/{split}_spearman': spearman}
    LOGGER.info(f"validation {split} finished, "
                f"pearson {pearson} ,"
                f"kendall {kendall} ,"
                f"spearman {spearman}")

    return val_log, spearman

def validate(model, val_loader, split='val'):
    val_loss = 0
    tot_score = 0
    n_ex = 0
    st = time()

    results = {}
    for i, batch in tqdm(enumerate(val_loader)):
        with torch.no_grad():
            loss, rank_scores = model(batch, compute_loss=True, compute_step_loss=False)
            sample_size = batch['sample_size']
            scores = rank_scores.contiguous().view(-1, sample_size).detach().cpu()
            preds = np.argsort(-scores.numpy())[:,0]
            loss = loss.mean()
            val_loss += loss.item()
            tot_score += (len(preds) - np.count_nonzero(preds))
            n_ex += len(preds)
        if i == 99:
            break
    del loss
    del rank_scores

    val_loss = sum(all_gather_list(val_loss))
    tot_score = sum(all_gather_list(tot_score))
    n_ex = sum(all_gather_list(n_ex))
    tot_time = time()-st
    #val_loss /= n_ex
    val_acc = tot_score / n_ex
    val_log = {f'valid/{split}_loss': val_loss,
               f'valid/{split}_acc': val_acc,
               f'valid/{split}_ex_per_s': n_ex/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"score: {val_acc*100:.2f} ,"
                f"loss: {val_loss:.3f}")
    return val_log, results

def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1]  # argmax
    one_hots = torch.zeros(*labels.size(), device=labels.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--train_txt_db",
                        default=None, type=str,
                        help="The input train corpus. (LMDB)")
    parser.add_argument("--train_img_db",
                        default=None, type=str,
                        help="The input train images.")
    parser.add_argument("--val_txt_db",
                        default=None, type=str,
                        help="The input validation corpus. (LMDB)")
    parser.add_argument("--val_img_db",
                        default=None, type=str,
                        help="The input validation images.")
    parser.add_argument("--test_txt_db",
                        default=None, type=str,
                        help="The input test corpus. (LMDB)")
    parser.add_argument("--test_img_db",
                        default=None, type=str,
                        help="The input test images.")
    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')
    parser.add_argument("--model_config",
                        default=None, type=str,
                        help="json file for model architecture")
    parser.add_argument("--checkpoint",
                        default="/pretrain/uniter-base.pt", type=str,
                        help="pretrained model (can take 'google-bert') ")

    parser.add_argument(
        "--output_dir", default=None, type=str,
        help="The output directory where the model checkpoints will be "
             "written.")

    # Prepro parameters
    parser.add_argument('--max_txt_len', type=int, default=60,
                        help='max number of tokens in text (BERT BPE)')
    parser.add_argument('--conf_th', type=float, default=0.2,
                        help='threshold for dynamic bounding boxes '
                             '(-1 for fixed)')
    parser.add_argument('--max_bb', type=int, default=100,
                        help='max number of bounding boxes')
    parser.add_argument('--min_bb', type=int, default=10,
                        help='min number of bounding boxes')
    parser.add_argument('--num_bb', type=int, default=36,
                        help='static number of bounding boxes')

    # training parameters
    parser.add_argument("--train_batch_size",
                        default=4096, type=int,
                        help="Total batch size for training. "
                             "(batch by tokens)")
    parser.add_argument("--val_batch_size",
                        default=4096, type=int,
                        help="Total batch size for validation. "
                             "(batch by tokens)")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=16,
                        help="Number of updates steps to accumualte before "
                             "performing a backward/update pass.")
    parser.add_argument("--margin", default=0.2, type=float,
                        help="margin of ranking loss")
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--valid_steps",
                        default=1000,
                        type=int,
                        help="Run validation every X steps")
    parser.add_argument("--num_train_steps",
                        default=100000,
                        type=int,
                        help="Total number of training updates to perform.")
    parser.add_argument("--optim", default='adam',
                        choices=['adam', 'adamax', 'adamw'],
                        help="optimizer")
    parser.add_argument("--betas", default=[0.9, 0.98], nargs='+',
                        help="beta for adam optimizer")
    parser.add_argument("--dropout",
                        default=0.1,
                        type=float,
                        help="tune dropout regularization")
    parser.add_argument("--weight_decay",
                        default=0.0,
                        type=float,
                        help="weight decay (L2) regularization")
    parser.add_argument("--grad_norm",
                        default=0.25,
                        type=float,
                        help="gradient clipping (-1 for no clipping)")
    parser.add_argument("--warmup_steps",
                        default=4000,
                        type=int,
                        help="Number of training steps to perform linear "
                             "learning rate warmup for.")

    # device parameters
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true',
                        help="pin memory")

    # can use config files
    parser.add_argument('--config', help='JSON config files')

    args = parse_with_config(parser)

    '''
    if exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not "
                         "empty.".format(args.output_dir))
    '''

    if args.conf_th == -1:
        assert args.max_bb + args.max_txt_len + 2 <= 512
    else:
        assert args.num_bb + args.max_txt_len + 2 <= 512

    main(args)
