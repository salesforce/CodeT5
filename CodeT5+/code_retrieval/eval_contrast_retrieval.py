'''
 * Copyright (c) 2023, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Yue Wang
'''
import argparse
import os
import pprint
import json
import time
import datetime
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from data_utils import create_dataset, create_loader


@torch.no_grad()
def get_feats(model, tokenizer, data_loader, max_length, device, desc='Get feats'):
    embeds = []

    for text in tqdm(data_loader, total=len(data_loader), desc=desc):
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=max_length,
                               return_tensors="pt").to(device)
        embed = model(text_input.input_ids, attention_mask=text_input.attention_mask)
        embeds.append(embed)

    embeds = torch.cat(embeds, dim=0)

    return embeds


@torch.no_grad()
def contrast_evaluation(text_embeds, code_embeds, img2txt):
    score_matrix_i2t = text_embeds @ code_embeds.t()
    scores_i2t = score_matrix_i2t.cpu().numpy()

    ranks = np.ones(scores_i2t.shape[0]) * -1
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == img2txt[index])[0][0]

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    mrr = 100.0 * np.mean(1 / (ranks + 1))

    eval_result = {'r1': tr1,
                   'r5': tr5,
                   'r10': tr10,
                   'mrr': mrr}
    return eval_result


def main(args):
    print("\nCreating retrieval dataset")
    _, _, test_dataset, code_dataset = create_dataset(args.data_dir, args.lang)

    test_loader, code_loader = create_loader([test_dataset, code_dataset], [None, None],
                                             batch_size=[args.batch_size, args.batch_size],
                                             num_workers=[4, 4], is_trains=[False, False], collate_fns=[None, None])

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True)
    print(f'Loaded {args.model_name} model (#para={model.num_parameters()})')

    print('\nStart zero-shot evaluation...')
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    text_embeds = get_feats(model, tokenizer, test_loader, args.max_text_len, device, desc='Get text feats')
    code_embeds = get_feats(model, tokenizer, code_loader, args.max_code_len, device, desc='Get code feats')
    test_result = contrast_evaluation(text_embeds, code_embeds, test_loader.dataset.text2code)
    print(f'\n====> zero-shot test result: ', test_result)

    if args.local_rank in [-1, 0]:
        log_stats = {
            **{f'test_{k}': v for k, v in test_result.items()},
            'epoch': -1,
        }

        with open(os.path.join(args.output_dir, "result.txt"), "a") as f:
            f.write(json.dumps(log_stats) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lang', type=str,
                        choices=['ruby', 'javascript', 'go', 'python', 'java', 'php', 'AdvTest', 'cosqa'])
    parser.add_argument('--model_name', type=str, default='Salesforce/codet5p-110m-embedding')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--max_text_len', default=64, type=int)
    parser.add_argument('--max_code_len', default=360, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--local_rank', default=-1, type=int)

    args = parser.parse_args()

    argsdict = vars(args)
    if args.local_rank in [0, -1]:
        print(pprint.pformat(argsdict))

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "command.txt"), 'w') as f:
        f.write(pprint.pformat(argsdict))

    main(args)
