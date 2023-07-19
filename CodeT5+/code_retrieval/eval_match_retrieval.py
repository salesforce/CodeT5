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
def get_feats(model, tokenizer, data_loader, max_length, device, modality='text', desc='Get feats'):
    text_ids = []
    text_embeds = []
    text_atts = []
    text_outputs = []

    for text in tqdm(data_loader, total=len(data_loader), desc=desc):
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=max_length,
                               return_tensors="pt").to(device)
        text_output = model.encoder(text_input.input_ids, attention_mask=text_input.attention_mask,
                                    return_dict=True)
        text_embed = torch.nn.functional.normalize(model.proj(text_output.last_hidden_state[:, 0, :]), dim=-1)

        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)
        text_embeds.append(text_embed)
        if modality == 'text':
            text_outputs.append(text_output.last_hidden_state.cpu())

    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    text_embeds = torch.cat(text_embeds, dim=0)
    if modality == 'text':
        text_outputs = torch.cat(text_outputs, dim=0)
    return text_ids, text_atts, text_embeds, text_outputs


def get_eos_vec(hidden_state, source_ids, eos_token_id):
    eos_mask = source_ids.eq(eos_token_id)
    if len(torch.unique(eos_mask.sum(1))) > 1:
        raise ValueError("All examples must have the same number of <eos> tokens.")
    dec_vec = hidden_state[eos_mask, :].view(hidden_state.size(0), -1, hidden_state.size(-1))[:, -1, :]
    return dec_vec


@torch.no_grad()
def match_evaluation(model, text_feats, code_feats, tokenizer, device, top_k, img2txt):
    start_time = time.time()

    text_ids, text_atts, text_embeds, text_outputs = text_feats
    code_ids, code_atts, code_embeds, _ = code_feats
    code_ids[:, 0] = tokenizer.enc_token_id

    sims_matrix = text_embeds @ code_embeds.t()
    score_matrix_i2t = torch.full((text_ids.size(0), code_ids.size(0)), -100.0).to(device)

    for i, sims in enumerate(tqdm(sims_matrix, desc=f'Evaluate text-code matching with top {top_k} candidates:')):
        topk_sim, topk_idx = sims.topk(k=top_k, dim=0)
        encoder_output = text_outputs[i].repeat(top_k, 1, 1).to(device)
        encoder_att = text_atts[i].repeat(top_k, 1).to(device)
        code_ids[:, 0] = tokenizer.enc_token_id
        output = model.decoder(code_ids[topk_idx],
                               attention_mask=code_atts[topk_idx],
                               encoder_hidden_states=encoder_output,
                               encoder_attention_mask=encoder_att,
                               return_dict=True,
                               )
        output_vec = get_eos_vec(output.last_hidden_state, code_ids[topk_idx], tokenizer.eos_token_id)
        score = model.itm_head(output_vec)[:, 1]
        score_matrix_i2t[i, topk_idx] = score + topk_sim

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))

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
    tokenizer.enc_token_id = tokenizer.convert_tokens_to_ids('[ENC]')
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True)
    print(f'Loaded {args.model_name} model (#para={model.num_parameters()})')

    print('\nStart zero-shot evaluation...')
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    text_feats = get_feats(model, tokenizer, test_loader, args.max_text_len, device, modality='text',
                           desc='Get text feats')
    code_feats = get_feats(model, tokenizer, code_loader, args.max_code_len, device, modality='code',
                           desc='Get code feats')

    test_result = match_evaluation(model, text_feats, code_feats, tokenizer, device, args.top_k,
                                   test_loader.dataset.text2code)
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
    parser.add_argument('--model_name', type=str, default='Salesforce/codet5p-220m-bimodal')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--top_k', default=256, type=int, help='top k candidates')
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
