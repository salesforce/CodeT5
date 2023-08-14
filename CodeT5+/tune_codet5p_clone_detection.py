# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function

import os
import json
import random
import logging
import argparse

import torch
import torch.nn as nn
import numpy as np


from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler

from transformers import (AdamW, get_linear_schedule_with_warmup,
                          AutoTokenizer, AutoModel)

logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_tokens,
                 input_ids,
                 attention_mask,
                 index,
                 label,
                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.index = index
        self.label = label


class Model(nn.Module):

    def __init__(self, encoder, tokenizer):
        super(Model, self).__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer

    def forward(self, input_ids, input_mask, p_input_ids, p_input_mask, n_input_ids, n_input_mask, labels=None):
        bs, _ = input_ids.size()
        # all_ids = torch.cat((input_ids, p_input_ids, n_input_ids), 0)
        all_mask = torch.cat((input_mask, p_input_mask, n_input_mask), 0)
        # outputs = self.encoder(all_ids, attention_mask=all_mask)[0]
        # outputs = (outputs * all_mask[:, :, None]).sum(1) / all_mask.sum(1)[:, None]
        # outputs = torch.nn.functional.normalize(outputs, p=2, dim=1)
        # outputs = outputs.split(bs, 0)

        input_embed = self.encoder(input_ids, attention_mask=input_mask)[0]
        p_embed = self.encoder(p_input_ids, attention_mask=p_input_mask)[0]
        n_embed = self.encoder(n_input_ids, attention_mask=n_input_mask)[0]

        # input_embed_n = torch.nn.functional.normalize()

        outputs = torch.cat((input_embed, p_embed, n_embed), 0)
        outputs = (outputs * all_mask[:, :, None]).sum(1) / all_mask.sum(1)[:, None]
        outputs = torch.nn.functional.normalize(outputs, p=2, dim=1)[0]
        outputs = outputs.split(len(input_embed), 0)

        input_embed_n = outputs[0].resize(1, len(input_embed))
        p_embed_n = outputs[1].resize(1, len(p_embed))
        n_embed_n = outputs[2].resize(1, len(n_embed))

        prob_1 = (input_embed_n * p_embed_n).sum(-1) * 20
        prob_2 = (input_embed_n * n_embed_n).sum(-1) * 20
        temp = torch.cat((input_embed_n, p_embed_n), 0)
        temp_labels = torch.cat((labels, labels), 0)
        prob_3 = torch.mm(input_embed_n, temp.t()) * 20
        mask = labels[:, None] == temp_labels[None, :]
        prob_3 = prob_3 * (1 - mask.float()) - 1e9 * mask.float()

        prob = torch.softmax(torch.cat((prob_1[:, None], prob_2[:, None], prob_3), -1), -1)
        loss = torch.log(prob[:, 0] + 1e-10)
        loss = -loss.mean()

        pdb.set_trace()

        return loss, input_embed_n


def convert_examples_to_features(js: dict, tokenizer, args):
    """convert examples to token ids"""
    text_input = tokenizer(js["code"], padding='max_length',
                           truncation=True, max_length=args.block_size, return_tensors="pt")
    source_ids = text_input.input_ids[0]
    source_tokens = tokenizer.convert_ids_to_tokens(text_input.input_ids[0])
    attention_mask = text_input.attention_mask[0]
    return InputFeatures(source_tokens, source_ids, attention_mask, js['index'], int(js['label']))


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        data = []
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                data.append(js)
        for js in data:
            self.examples.append(convert_examples_to_features(js, tokenizer, args))
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("label: {}".format(example.label))
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
        self.label_examples = {}
        for e in self.examples:
            if e.label not in self.label_examples:
                self.label_examples[e.label] = []
            self.label_examples[e.label].append(e)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        label = self.examples[i].label
        index = self.examples[i].index
        labels = list(self.label_examples)
        labels.remove(label)
        while True:
            shuffle_example = random.sample(self.label_examples[label], 1)[0]
            if shuffle_example.index != index:
                p_example = shuffle_example
                break
        n_example = random.sample(self.label_examples[random.sample(labels, 1)[0]], 1)[0]

        return (
            self.examples[i].input_ids, self.examples[i].attention_mask,
            p_example.input_ids, p_example.attention_mask,
            n_example.input_ids, n_example.attention_mask,
            torch.tensor(label)
        )


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size, num_workers=4, pin_memory=True)

    args.max_steps = args.num_train_epochs * len(train_dataloader)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps * 0.1,
                                                num_training_steps=args.max_steps)

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}", )
    logger.info(f"  Num Epochs = {args.num_train_epochs}", )
    logger.info(f"  Instantaneous batch size per GPU = {args.train_batch_size // args.n_gpu}")
    logger.info(f"  Total train batch size = {args.train_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_steps}")

    losses, best_map = [], 0

    model.zero_grad()
    for idx in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            inputs_ids = batch[0].to(args.device)
            inputs_mask = batch[1].to(args.device)
            p_inputs_ids = batch[2].to(args.device)
            p_inputs_mask = batch[3].to(args.device)
            n_inputs_ids = batch[4].to(args.device)
            n_inputs_mask = batch[5].to(args.device)
            labels = batch[6].to(args.device)
            model.train()
            loss, vec = model(inputs_ids, inputs_mask,
                              p_inputs_ids, p_inputs_mask,
                              n_inputs_ids, n_inputs_mask,
                              labels)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            losses.append(loss.item())

            if (step + 1) % 100 == 0:
                logger.info("epoch {} step {} loss {}".format(idx, step + 1, round(np.mean(losses[-100:]), 4)))

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        results = evaluate(args, model, tokenizer, args.eval_data_file)
        for key, value in results.items():
            logger.info(f"  {key} = {round(value, 4)}")

        if results['eval_map'] > best_map:
            best_map = results['eval_map']
            logger.info("  " + "*" * 20)
            logger.info(f"  Best map:{round(best_map, 4)}")
            logger.info("  " + "*" * 20)

            checkpoint_prefix = 'checkpoint-best-map'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info(f"Saving model checkpoint to {output_dir}")


def evaluate(args, model, tokenizer, data_file):
    """ Evaluate the model """
    eval_dataset = TextDataset(tokenizer, args, data_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4)

    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Batch size = {args.eval_batch_size}")
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    vecs = []
    labels = []
    for batch in eval_dataloader:
        inputs_ids = batch[0].to(args.device)
        inputs_mask = batch[1].to(args.device)
        p_inputs_ids = batch[2].to(args.device)
        p_inputs_mask = batch[3].to(args.device)
        n_inputs_ids = batch[4].to(args.device)
        n_inputs_mask = batch[5].to(args.device)
        tmp_labels = batch[6].to(args.device)
        with torch.no_grad():
            lm_loss, vec = model(inputs_ids, inputs_mask,
                                 p_inputs_ids, p_inputs_mask,
                                 n_inputs_ids, n_inputs_mask,
                                 tmp_labels)
            eval_loss += lm_loss.mean().item()
            vecs.append(vec.cpu().numpy())
            labels.append(tmp_labels.cpu().numpy())
        nb_eval_steps += 1
    vecs = np.concatenate(vecs, 0)
    labels = np.concatenate(labels, 0)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)

    scores = np.matmul(vecs, vecs.T)
    dic = {}
    for i in range(scores.shape[0]):
        scores[i, i] = -1000000
        if int(labels[i]) not in dic:
            dic[int(labels[i])] = -1
        dic[int(labels[i])] += 1
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]
    MAP = []
    for i in range(scores.shape[0]):
        cont = 0
        label = int(labels[i])
        Avep = []
        for j in range(dic[label]):
            index = sort_ids[i, j]
            if int(labels[index]) == label:
                Avep.append((len(Avep) + 1) / (j + 1))
        MAP.append(sum(Avep) / dic[label])

    result = {
        "eval_loss": float(perplexity),
        "eval_map": float(np.mean(MAP))
    }

    return result


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--train_data_file", default=None, type=str,
                        help="The input training data file (a jsonl file).")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a jsonl file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to evaluate the perplexity on (a jsonl file).")
    parser.add_argument('--cache', type=str, default='cache')
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    # print arguments
    args = parser.parse_args()
    # set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info(f"device: {device}, n_gpu: {args.n_gpu}")

    # Set seed
    set_seed(args.seed)

    # build model
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-110m-embedding",
                                              cache_dir=args.cache, trust_remote_code=True)
    model = AutoModel.from_pretrained("Salesforce/codet5p-110m-embedding",
                                      cache_dir=args.cache, trust_remote_code=True)

    model = Model(model, tokenizer)
    logger.info(f"Training/evaluation parameters {args}")

    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

        # Training
    if args.do_train:
        train_dataset = TextDataset(tokenizer, args, args.train_data_file)
        train(args, train_dataset, model, tokenizer)

    # Evaluation
    results = {}
    if args.do_eval:
        checkpoint_prefix = 'checkpoint-best-map/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model_to_load = model.module if hasattr(model, 'module') else model
        model_to_load.load_state_dict(torch.load(output_dir))
        result = evaluate(args, model, tokenizer, args.eval_data_file)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info(f"  {key} = {str(round(result[key] * 100 if 'map' in key else result[key], 2))}")

    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-map/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model_to_load = model.module if hasattr(model, 'module') else model
        model_to_load.load_state_dict(torch.load(output_dir))
        result = evaluate(args, model, tokenizer, args.test_data_file)
        logger.info("***** Test results *****")
        for key in sorted(result.keys()):
            logger.info(f"  {key} = {str(round(result[key] * 100 if 'map' in key else result[key], 2))}")


if __name__ == "__main__":
    main()

