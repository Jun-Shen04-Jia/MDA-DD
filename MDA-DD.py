from collections import defaultdict
import glob
import os
import argparse
import re
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision.utils
import math
import random
import wandb
import copy
import datetime
import time
import torch.nn.functional as F
from data import get_dataset_flickr
from src.epoch import evaluate_synset
from src.networks import CLIPModel_full
from src.utils import ParamDiffAug, get_time, TensorDataset
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
from torch.utils.data import DataLoader, Subset

def make_timestamp(prefix: str="", suffix: str="") -> str:
    tmstamp = '{:%m%d_%H%M%S}'.format(datetime.datetime.now())
    return prefix + tmstamp + suffix

def get_images_texts(n, dataset, args, text_encoder, seed=None, get_text_raw=False):
    if seed != None:
        np.random.seed(seed)
    idx_shuffle = np.random.permutation(len(dataset))[:n]

    # Initialize the text encoder
    with torch.no_grad():
        text_encoder.eval()

        image_syn = torch.stack([dataset[i][0] for i in idx_shuffle])
        texts = [dataset[i][1] for i in idx_shuffle]

        encoding = text_encoder.tokenizer.batch_encode_plus(texts, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoding['input_ids'].to(args.device)
        attention_mask = encoding['attention_mask'].to(args.device)

        text_syn = text_encoder.model.embeddings(
            input_ids=input_ids,
        )
    
    if get_text_raw:
        return image_syn, text_syn.float(), attention_mask, texts
    else:
        return image_syn, text_syn.float(), attention_mask

def main(args):
    ''' organize the real train dataset '''  
    trainloader, testloader, train_dataset, test_dataset = get_dataset_flickr(args)
    train_iter = iter(trainloader)

    def get_batch_real(net, train_iter):
        try:
            images, text, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(trainloader)
            images, text, _ = next(train_iter)
        
        if args.text_encoder in ['bert', 'distilbert']:
            encoding = net.text_encoder.tokenizer.batch_encode_plus(text, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoding['input_ids'].to(args.device)
            attention_mask = encoding['attention_mask'].to(args.device)

            text_emb = net.text_encoder.model.embeddings(
                input_ids=input_ids,
            )
            return images, text_emb, attention_mask, train_iter
        elif args.text_encoder == 'clip':
            tokens = clip.tokenize(text).cuda()
            text_emb = net.text_encoder.model.token_embedding(tokens)
            return images, text_emb, tokens, train_iter
    realloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size_train,
        num_workers=2,
        pin_memory=True,
        sampler=None,
        shuffle=True,
        collate_fn=None,
        drop_last=True,
    )

    real_iter = iter(realloader)

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))
    print('Hyper-parameters: \n', args.__dict__)

    if args.eval_it > 0:
        eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    else:
        eval_it_pool = []
    if args.wandb:
        wandb.init(
            project="MDA-DD",
            name=f"main_{args.dataset}_N{args.num_queries}",
            config=args
        )
    else:
        wandb.init(mode = 'disabled')
    ''' initialize the synthetic data '''
    net = CLIPModel_full(args).to('cuda')
    net.eval()

    image_syn, text_syn, mask_syn = get_images_texts(args.num_queries, train_dataset, args,net.text_encoder, seed=args.seed)

    del net
    torch.cuda.empty_cache()

    ''' training params setup '''
    image_syn = image_syn.detach().to(args.device).requires_grad_(True)
    text_syn = text_syn.detach().to(args.device).requires_grad_(True)

    optimizer = torch.optim.SGD([
        {'params': [image_syn], 'lr': args.lr_img, "momentum": args.momentum_syn},
        {'params': [text_syn], 'lr': args.lr_txt, "momentum": args.momentum_syn},
    ], lr=0)
    optimizer.zero_grad()
    stagger_interval = args.max_model_lifetime // args.num_models_in_queue  
    def create_expert_model():
        net = CLIPModel_full(args).to('cuda')
        opt = torch.optim.SGD([
            {'params': net.image_encoder.parameters(), 'lr': args.lr_encoder_img},
            {'params': net.image_projection.parameters(), 'lr': args.lr_proj_img},
            {'params': net.text_encoder.parameters(), 'lr': args.lr_encoder_txt},
            {'params': net.text_projection.parameters(), 'lr': args.lr_proj_txt},
        ], lr=0, momentum=0.9, weight_decay=0.0005)
        return {"net": net, "opt": opt, "step": 0}

    model_queue = [create_expert_model()]
    # =========================================================

    for it in tqdm(range(args.Iteration + 1)):
        ''' Evaluate synthetic data '''
        if it in eval_it_pool :
            print('Evaluation\nimage_model_train = %s, text_model_train = %s, iteration = %d'%(args.image_encoder, args.text_encoder, it))

            multi_eval_aggr_result = defaultdict(list)
            for it_eval in range(args.num_eval):
                net_eval = CLIPModel_full(args)
                image_syn_eval = copy.deepcopy(image_syn.detach())
                text_syn_eval = copy.deepcopy(text_syn.detach())
                mask_syn_eval = copy.deepcopy(mask_syn.detach())

                _, _, best_val_result = evaluate_synset(it_eval, net_eval, image_syn_eval, text_syn_eval, mask_syn_eval, testloader, test_dataset, args)

                for k, v in best_val_result.items():
                    multi_eval_aggr_result[k].append(v)

            for key, values in multi_eval_aggr_result.items():
                print(f'{key}: {np.mean(values):.2f} ({np.std(values):.2f})')
                if key not in ["img_r_mean", "txt_r_mean"]:
                    wandb.log({
                        "Mean/{}".format(key): np.mean(values), 
                        "Std/{}".format(key): np.std(values)
                    })

            print(f'{np.mean(multi_eval_aggr_result["img_r1"]):.2f}\t{np.mean(multi_eval_aggr_result["img_r5"]):.2f}\t{np.mean(multi_eval_aggr_result["img_r10"]):.2f}\t{np.mean(multi_eval_aggr_result["txt_r1"]):.2f}\t{np.mean(multi_eval_aggr_result["txt_r5"]):.2f}\t{np.mean(multi_eval_aggr_result["txt_r10"]):.2f}\t{np.mean(multi_eval_aggr_result["r_mean"]):.2f}')

        ''' Save synthetic data '''
        if it in eval_it_pool:
            torch.cuda.empty_cache()
            with torch.no_grad():
                save_dir = os.path.join(args.logged_files, args.dataset, f'N{args.num_queries}')
                print("Saving to {}".format(save_dir))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                image_save = image_syn.detach().cpu()
                text_save = text_syn.detach().cpu()
                mask_save = mask_syn.detach().cpu()

        for m_dict in model_queue:
            expert_net = m_dict["net"]
            optimizer_net = m_dict["opt"]
            step = m_dict["step"]

            expert_net.eval()
            with torch.no_grad():
                image_real, text_real, mask_real, train_iter = get_batch_real(expert_net, train_iter)
                    
                image_real = image_real.to(args.device).detach()
                text_real = text_real.to(args.device).detach()

                img_embed_real = expert_net.image_encoder(image_real).float()
                img_feat_real = expert_net.image_projection(img_embed_real)

                txt_embed_real = expert_net.text_encoder(text_real, mask_real).float()
                txt_feat_real = expert_net.text_projection(txt_embed_real)

                socm_real = (img_embed_real.T @ txt_embed_real) / img_embed_real.shape[0]

            if args.num_queries > args.batch_syn:
                idx_batch = np.random.permutation(args.num_queries)[:args.batch_syn]
                image_syn_batch = image_syn[idx_batch]
                text_syn_batch = text_syn[idx_batch]
                mask_syn_batch = mask_syn[idx_batch]
            else:
                image_syn_batch, text_syn_batch, mask_syn_batch = image_syn, text_syn, mask_syn

            img_embed_syn = expert_net.image_encoder(image_syn_batch).float()
            img_feat_syn = expert_net.image_projection(img_embed_syn)

            txt_embed_syn = expert_net.text_encoder(text_syn_batch, mask_syn_batch).float()
            txt_feat_syn = expert_net.text_projection(txt_embed_syn)

            socm_syn = (img_embed_syn.T @ txt_embed_syn) / img_embed_syn.shape[0]

            loss_socm = torch.norm(socm_syn - socm_real * args.rho, p='fro')**2
            loss_img_feat = torch.sum((torch.mean(img_feat_real, dim=0) - torch.mean(img_feat_syn, dim=0)) ** 2)
            loss_txt_feat = torch.sum((torch.mean(txt_feat_real, dim=0) - torch.mean(txt_feat_syn, dim=0)) ** 2)

            total_loss = loss_socm + args.lamda * (loss_img_feat + loss_txt_feat)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

                
            if m_dict is model_queue[0]:
                norm_img = torch.mean(torch.linalg.norm(image_syn.view(image_syn.shape[0], -1), dim=1))
                norm_txt = torch.mean(torch.linalg.norm(text_syn, dim=1))
                    
                wandb.log({
                    "Loss/total_loss": total_loss.item(),
                    "Loss/loss_socm": loss_socm.item(),
                    "Loss/loss_img_feat": loss_img_feat.item(),
                    "Loss/loss_txt_feat": loss_txt_feat.item(),
                    "Norm/norm_img": norm_img.item(),
                    "Norm/norm_txt": norm_txt.item(),
                })
                    
                if it % 10 == 0:
                    print('%s iter = %05d, M0_step = %04d, total_loss = %.4f, loss_socm = %.4f, norm_img = %.4f, norm_txt = %.4f' % (get_time(), it, step, total_loss.item(), loss_socm.item(), norm_img.item(), norm_txt.item()))

            
            expert_net.train()
            loss_train, acc_train, num_exp = 0, 0, 0
            
            for i in range(args.inner_loop):
                try:
                    image, text_raw, _ = next(real_iter)
                except StopIteration:
                    real_iter = iter(realloader)
                    image, text_raw, _ = next(real_iter)

                with torch.no_grad():
                    encoding = expert_net.text_encoder.tokenizer.batch_encode_plus(text_raw, return_tensors='pt', padding=True, truncation=True)
                    input_ids = encoding['input_ids'].to(args.device)
                    mask = encoding['attention_mask'].to(args.device)
                    text = expert_net.text_encoder.model.embeddings(input_ids=input_ids)

                image = image.to(args.device)
                n_b = image.shape[0]

                loss, acc = expert_net(image, text, mask)

                loss_train += loss.item() * n_b
                acc_train += acc
                num_exp += n_b

                optimizer_net.zero_grad()
                loss.backward()
                optimizer_net.step()

           
            m_dict["step"] += 1

       
        popped_this_turn = False
        if model_queue[0]["step"] >= args.max_model_lifetime:
            old_model = model_queue.pop(0)
            del old_model["net"]
            del old_model["opt"]
            torch.cuda.empty_cache()
            popped_this_turn = True

        if len(model_queue) < args.num_models_in_queue:
            if popped_this_turn or model_queue[-1]["step"] >= stagger_interval:
                model_queue.append(create_expert_model())

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    # main
    parser.add_argument('--dataset', type=str, default='flickr', help='dataset')
    parser.add_argument('--num_queries', type=int, default=100, help='number of queries')
    parser.add_argument('--rho', type=float, default=1.0, help='scaling factor for real cross-covariance')
    parser.add_argument('--lamda', type=float, default=1.0, help='weight for feature matching loss')
    parser.add_argument('--num_models_in_queue', type=int, default=8, help='num_models_in_queue')
    parser.add_argument('--max_model_lifetime', type=int, default=200, help='max_model_lifetime')

    # network
    parser.add_argument('--image_encoder', type=str, default='nfnet',  help='image encoder')
    parser.add_argument('--text_encoder', type=str, default='bert', help='text encoder')
    parser.add_argument('--image_pretrained', type=bool, default=True, help='image_pretrained')
    parser.add_argument('--text_pretrained', type=bool, default=True, help='text_pretrained')
    parser.add_argument('--image_trainable', type=bool, default=True, help='image_trainable')
    parser.add_argument('--text_trainable', type=bool, default=True, help='text_trainable')
    parser.add_argument('--proj_dim', type=int, default=2304, help='projection dimension')

    # data
    parser.add_argument('--image_size', type=int, default=224, help='image_size')
    parser.add_argument('--ann_root', type=str, default='./data/Flickr30k_ann/', help='location of ann root')
    parser.add_argument('--image_root', type=str, default='distill_utils/data/Flickr30k/', help='location of image root')

    # distillation
    parser.add_argument('--Iteration', type=int, default=1000, help='how many global distillation steps to perform')
    parser.add_argument('--inner_loop', type=int, default=1, help='number of training steps for one online model update')

    parser.add_argument('--batch_size_train', type=int, default=128, help='batch_size_train (for real)')
    parser.add_argument('--batch_syn', type=int, default=200, help='batch_syn')

    parser.add_argument('--lr_img', type=float, default=1, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_txt', type=float, default=1, help='learning rate for updating synthetic texts')
    parser.add_argument('--momentum_syn', type=float, default=0.5)

    # evaluation
    parser.add_argument('--eval_it', type=int, default=200, help='how often to evaluate')
    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')
    parser.add_argument('--epoch_eval_train', type=int, default=100, help='epochs to train a model with synthetic data')
    parser.add_argument('--batch_size_test', type=int, default=128, help='batch_size_test')
    parser.add_argument('--lr_encoder_img', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--lr_encoder_txt', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--lr_proj_img', type=float, default=0.1, help='learning rate for updating network parameters')
    parser.add_argument('--lr_proj_txt', type=float, default=0.1, help='learning rate for updating network parameters')

    # etc
    parser.add_argument('--wandb', action="store_true", help='wandb')
    parser.add_argument('--save', action="store_true", help='save')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--logged_files', type=str, default='results', help='path to save synthetic dataset')
    
    args = parser.parse_args()

    if args.dataset == 'flickr':
        args.image_root = './distill_utils/data/Flickr30k/'
    elif args.dataset == 'coco':
        args.image_root = './distill_utils/data/COCO/'

    main(args)