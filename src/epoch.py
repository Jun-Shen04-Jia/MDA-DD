from math import ceil
import numpy as np
import torch
import time
import datetime
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
from src.utils import *

import argparse
from data import get_dataset_flickr
from src.networks import CLIPModel_full
import copy
from src.vl_distill_utils import get_images_texts



def epoch(e, dataloader, net, optimizer, args):
    net = net.to(args.device)
    net.train()
    loss_avg, acc_avg, num_exp = 0, 0, 0

    for i, data in (enumerate(dataloader)):
        image, text, mask = data

        image = image.to(args.device)
        n_b = image.shape[0]

        loss, acc = net(image, text, mask)

        loss_avg += loss.item() * n_b
        acc_avg += acc
        num_exp += n_b

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg



@torch.no_grad()
def epoch_test(dataloader, testset, model, device, args):

    model.eval()

    logit_scale = model.logit_scale.detach().exp()

    start_time = time.time()

    batch_size = 1000

    text_raw_list = []

    text_proj_list = []

   

    with torch.no_grad():

        for i in range(0, len(testset.text), batch_size):

            batch_texts = testset.text[i:i+batch_size]


            raw_output = model.text_encoder.forward_raw(batch_texts).float()

            proj_output = model.text_projection(raw_output)


            text_proj_norm = proj_output / proj_output.norm(dim=1, keepdim=True)


            text_raw_norm = raw_output / raw_output.norm(dim=1, keepdim=True)

           

            text_raw_list.append(text_raw_norm)

            text_proj_list.append(text_proj_norm)



    text_raw_all = torch.cat(text_raw_list, dim=0).to(device)

    text_proj_all = torch.cat(text_proj_list, dim=0).to(device)



    image_raw_list = []

    image_proj_list = []

   

    with torch.no_grad():

        for image, img_id in dataloader:

            image = image.to(device)


            raw_feat = model.image_encoder(image).float()


            proj_feat = model.image_projection(raw_feat)


            image_raw_norm = raw_feat / raw_feat.norm(dim=1, keepdim=True)


            image_proj_norm = proj_feat / proj_feat.norm(dim=1, keepdim=True)

           

            image_raw_list.append(image_raw_norm)

            image_proj_list.append(image_proj_norm)



    image_raw_all = torch.cat(image_raw_list, dim=0)

    image_proj_all = torch.cat(image_proj_list, dim=0)




    sims_t2i = logit_scale * image_raw_all @ text_proj_all.t()

   

    sims_i2t = logit_scale * image_proj_all @ text_raw_all.t()

   

    sims_matrix = (sims_t2i + sims_i2t) / 2.0




    # Image-to-Text

    score_matrix_i2t = torch.full((len(image_raw_all), len(text_raw_all)), -100.0).to(device)

    for i, sims in enumerate(sims_matrix):

        topk_sim, topk_idx = sims.topk(k=min(128, sims.size(0)), dim=0)

        score_matrix_i2t[i, topk_idx] = topk_sim

   

    # Text-to-Image

    sims_matrix_t = sims_matrix.t()

    score_matrix_t2i = torch.full((len(text_raw_all), len(image_raw_all)), -100.0).to(device)

    for i, sims in enumerate(sims_matrix_t):

        topk_sim, topk_idx = sims.topk(k=min(128, sims.size(0)), dim=0)

        score_matrix_t2i[i, topk_idx] = topk_sim



    total_time = time.time() - start_time

    print(f'Evaluation time {str(datetime.timedelta(seconds=int(total_time)))}')



    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()



@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    
    #Images->Text 
    ranks = np.zeros(scores_i2t.shape[0])
    # print("TR: ", len(ranks))
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
  
    #Text->Images 
    ranks = np.zeros(scores_t2i.shape[0])
    # print("IR: ", len(ranks))
    
    for index,score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)        

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result =  {'txt_r1': tr1,
                    'txt_r5': tr5,
                    'txt_r10': tr10,
                    'txt_r_mean': tr_mean,
                    'img_r1': ir1,
                    'img_r5': ir5,
                    'img_r10': ir10,
                    'img_r_mean': ir_mean,
                    'r_mean': r_mean}
    return eval_result


def evaluate_synset(it_eval, net, images_train, texts_train, mask_train, testloader, test_dataset, args, return_loss=False):
    
    net = net.to(args.device)
    images_train = images_train.to(args.device)
    texts_train = texts_train.to(args.device)
    mask_train = mask_train.to(args.device)
    Epoch = int(args.epoch_eval_train)

    optimizer = torch.optim.SGD([
        {'params': net.image_encoder.parameters(), 'lr': args.lr_encoder_img},
        {'params': net.image_projection.parameters(), 'lr': args.lr_proj_img},
        {'params': net.text_encoder.parameters(), 'lr': args.lr_encoder_txt},
        {'params': net.text_projection.parameters(), 'lr': args.lr_proj_txt},
    ], lr=0, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[Epoch//2+1], gamma=0.1)

    dst_train = TensorDataset(images_train, texts_train, mask_train)
    train_loader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_size_train, shuffle=True, num_workers=0)

    start = time.time()
    acc_train_list = []
    loss_train_list = []

    eval_epochs = [Epoch]
    eval_freq = Epoch // 10
    eval_epochs = list(range(0, Epoch+1, eval_freq)) + [Epoch] # eval 10 times in total

    best_val_result = None
    for ep in tqdm(range(Epoch+1), ncols=60):
        loss_train, acc_train = epoch(ep, train_loader, net, optimizer, args)
        acc_train_list.append(acc_train)
        loss_train_list.append(loss_train)
        if ep in eval_epochs:
            with torch.no_grad():
                score_val_i2t, score_val_t2i = epoch_test(testloader, test_dataset, net, args.device, args)
                val_result = itm_eval(score_val_i2t, score_val_t2i, testloader.dataset.txt2img, testloader.dataset.img2txt) 
                
                print("[Eval_{it:02d}] Ep{ep} | Image R@1={img_r1:.2f} R@5={img_r5:.2f} R@10={img_r10:.2f} | Text R@1={txt_r1:.2f} R@5={txt_r5:.2f} R@10={txt_r10:.2f} | Mean={r_mean:.2f}".format(
                    it=it_eval, ep=ep, **val_result
                ))
                if best_val_result is None or val_result["r_mean"] > best_val_result["r_mean"]:
                    best_val_result = val_result
                    
        lr_scheduler.step()

    time_train = time.time() - start
    return net, acc_train_list, best_val_result


