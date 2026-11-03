"""Move some basic utils in distill.py in VL-Distill here"""
import os
import numpy as np
import copy
import torch
import time
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from src.networks import TextEncoder, ImageEncoder
from tqdm import tqdm

import clip


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

