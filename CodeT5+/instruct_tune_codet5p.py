"""
Finetune CodeT5+ models on instruction tuning data
You can customize your own training data by following the HF dataset format to cache it to args.cache_data
Author: Yue Wang
Date: June 2023
"""

import os
import pprint
import argparse
import numpy as np
import copy
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e+6))


def freeze_decoder_except_xattn_codegen(model):
    print(f'Para before freezing: {model.num_parameters()}, trainable para: {get_model_size(model)}')
    for param in model.decoder.parameters():
        param.requires_grad = False

    num_decoder_layers = model.decoder.config.n_layer
    for i in range(num_decoder_layers):
        each_decoder_layer = model.decoder.transformer.h[i]
        if hasattr(each_decoder_layer, 'crossattention'):
            for param in each_decoder_layer.crossattention.parameters():
                param.requires_grad = True
            each_decoder_layer.crossattention.to(torch.float32)

        if hasattr(each_decoder_layer, 'alpha_xattn'):
            each_decoder_layer.alpha_xattn.requires_grad = True
    print(f'Para after freezing: {model.num_parameters()}, trainable para: {get_model_size(model)}')


def run_training(args, model, train_data):
    print(f"Starting main loop")

    training_args = TrainingArguments(
        report_to='tensorboard',
        output_dir=args.save_dir,
        overwrite_output_dir=False,

        do_train=True,
        save_strategy='epoch',

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_per_replica,
        gradient_accumulation_steps=args.grad_acc_steps,

        learning_rate=args.lr,
        weight_decay=0.0,
        warmup_steps=args.lr_warmup_steps,

        logging_dir=args.save_dir,
        logging_first_step=True,
        logging_steps=args.log_freq,
        save_total_limit=2,

        dataloader_drop_last=True,
        dataloader_num_workers=4,

        local_rank=args.local_rank,
        deepspeed=args.deepspeed,
        fp16=args.fp16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
    )

    trainer.train()

    if args.local_rank in [0, -1]:
        final_checkpoint_dir = os.path.join(args.save_dir, "final_checkpoint")
        model.save_pretrained(final_checkpoint_dir)
        print(f'  ==> Finish training and save to {final_checkpoint_dir}')


def load_tokenize_data(args):
    # Load and tokenize data
    if os.path.exists(args.cache_data):
        train_data = load_from_disk(args.cache_data)
        print(f'  ==> Loaded {len(train_data)} samples')
        return train_data
    else:
        datasets = load_dataset('json', data_files=args.instruct_data_path)['train']
        tokenizer = AutoTokenizer.from_pretrained(args.load)

        def preprocess_function(examples):
            prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
            source = [prompt_input.format_map({'instruction': instruct, 'input': inp}) if inp != ''
                      else prompt_no_input.format_map({'instruction': instruct})
                      for instruct, inp in zip(examples["instruction"], examples["input"])]
            target = [src + output + tokenizer.eos_token for src, output in zip(source, examples["output"])]

            model_inputs = tokenizer(source, max_length=args.max_len, padding="max_length", truncation=True)
            labels = tokenizer(target, max_length=args.max_len, padding="max_length", truncation=True)
            model_inputs["decoder_input_ids"] = copy.deepcopy(labels["input_ids"])

            # changing labels: convert all tokens in the duplicate prefix prompt and the padding part to -100
            eos_token_id = tokenizer.eos_token_id
            for x, y in zip(model_inputs["input_ids"], labels["input_ids"]):
                label_prefix_len = x.index(eos_token_id) if eos_token_id in x else len(x)
                y[:label_prefix_len] = [-100] * label_prefix_len

                if eos_token_id in y:
                    pad_len = len(y) - y.index(eos_token_id) - 1
                    if pad_len > 0:
                        y[y.index(eos_token_id) + 1:] = [-100] * pad_len

            # shift labels to the right as the decoder input and add decoder start token id
            decoder_start_id = tokenizer.eos_token_id
            for z in model_inputs["decoder_input_ids"]:
                z[1:] = z[:-1]
                z[0] = decoder_start_id

            model_inputs["labels"] = copy.deepcopy(labels["input_ids"])
            model_inputs["decoder_attention_mask"] = labels["attention_mask"]
            return model_inputs

        train_data = datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=datasets.column_names,
            num_proc=64,
            load_from_cache_file=False,
        )

        print(f'  ==> Loaded {len(train_data)} samples')
        train_data.save_to_disk(args.cache_data)
        print(f'  ==> Saved to {args.cache_data}')
        return train_data


def main(args):
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    # Save command to file
    with open(os.path.join(args.save_dir, "command.txt"), 'w') as f:
        f.write(pprint.pformat(argsdict))

    # Load and tokenize data using the tokenizer from `args.load`. If the data is already cached, load it from there.
    # You can customize this function to load your own data for any Seq2Seq LM tasks.
    train_data = load_tokenize_data(args)

    if args.data_num != -1:
        train_data = train_data.select([i for i in range(args.data_num)])

    # Load model from `args.load`
    model = AutoModelForSeq2SeqLM.from_pretrained(args.load, torch_dtype=torch.float16,
                                                  low_cpu_mem_usage=True, trust_remote_code=True)

    print(f"  ==> Loaded model from {args.load}, model size {model.num_parameters()}")
    freeze_decoder_except_xattn_codegen(model)

    run_training(args, model, train_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CodeT5+ instruction tuning")
    parser.add_argument('--data-num', default=-1, type=int)
    parser.add_argument('--max-len', default=512, type=int)
    parser.add_argument('--instruct-data-path', default='code_alpaca_20k.json', type=str)
    parser.add_argument('--cache-data', default='cache_data/instructions', type=str)
    parser.add_argument('--load', default='Salesforce/codet5p-16b', type=str)

    # Training
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--lr-warmup-steps', default=30, type=int)
    parser.add_argument('--batch-size-per-replica', default=1, type=int)
    parser.add_argument('--grad-acc-steps', default=16, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--deepspeed', default=None, type=str)
    parser.add_argument('--fp16', default=False, action='store_true')

    # Logging and stuff
    parser.add_argument('--save-dir', default="saved_models/instruct_codet5p_16b", type=str)
    parser.add_argument('--log-freq', default=10, type=int)
    parser.add_argument('--save-freq', default=500, type=int)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    main(args)
