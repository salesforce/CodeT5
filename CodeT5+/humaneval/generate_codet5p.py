import argparse
import pprint
import os
import re
from tqdm import tqdm
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from human_eval.data import write_jsonl, read_problems, stream_jsonl


def extract_text(prompt, remove_lines=True):
    token = '\"\"\"'
    start = token
    end = '>>>'

    start_idx = prompt.find(start) + len(start)
    end_idx = prompt.find(end)

    output = prompt[start_idx: end_idx]
    if remove_lines:
        output = output.replace('\n', ' ')
    output = re.sub(r"\s+", " ", output).strip()

    return output


INSTRUCTION = """Below is an instruction that describes a task. Write a response that appropriately completes the request.


### Instruction:
Create a Python script for this problem:
{}

### Response:"""


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='Salesforce/instructcodet5p-16b', help="")
    parser.add_argument('--output_path', type=str, help="")
    parser.add_argument('--start_index', type=int, default=0, help="")
    parser.add_argument('--end_index', type=int, default=164, help="")
    parser.add_argument('--temperature', type=float, default=0.8, help="")
    parser.add_argument('--N', type=int, default=200, help="")
    parser.add_argument('--max_len', type=int, default=600, help="")
    parser.add_argument('--decoding_style', type=str, default='sampling', help="")
    parser.add_argument('--num_seqs_per_iter', type=int, default=50, help='')
    parser.add_argument('--overwrite', action='store_true', help='')

    args = parser.parse_args()

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    STOP_SEQS = ['\nclass', '\ndef', '\n#', '\nif', '\nprint']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    problems = read_problems()

    task_ids = sorted(problems.keys())[args.start_index: args.end_index]
    prompts = [problems[task_id]['prompt'] for task_id in task_ids]
    num_samples = len(prompts)
    print("Number of samples: {}".format(num_samples))

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model,
                                                  trust_remote_code=True,  # False for 220m and 770m models
                                                  torch_dtype=torch.float16,
                                                  low_cpu_mem_usage=True)
    model.eval()
    model.to(device)

    # for larger LLMs such as 2B, 6B, and 16B, we need to pass the text prompt to the decoder
    prompt_to_decoder = True if any([size in args.model for size in ['2b', '6b', '16b']]) else False

    print(f"Loaded {args.model}.")
    for i in tqdm(range(num_samples), ncols=0, total=num_samples):
        output_file = args.output_path + '/{}.jsonl'.format(args.start_index + i)

        if os.path.exists(output_file) and not args.overwrite:
            print(f'Skip {output_file} as it already exists')
            continue

        prompt = prompts[i].replace('    ', '\t')
        if args.model == 'Salesforce/instructcodet5p-16b':
            prompt_batch = [INSTRUCTION.format(extract_text(prompt))]
            prompt_batch_decoder = [INSTRUCTION.format(extract_text(prompt)) + prompt]
        else:
            prompt_batch = [prompt]
            prompt_batch_decoder = [prompt]

        ids_batch = [task_ids[i]]

        completion_seqs = []

        encoding = tokenizer(prompt_batch, return_tensors="pt", truncation=True, max_length=args.max_len).to(device)
        encoding_decoder = tokenizer(prompt_batch_decoder, return_tensors="pt", truncation=True,
                                     max_length=args.max_len).to(device)

        if args.decoding_style == 'sampling':
            loops = int(args.N / args.num_seqs_per_iter)
        else:
            loops = 1

        for _ in tqdm(range(loops), total=loops, leave=False, ncols=0):

            with torch.no_grad():
                if args.decoding_style == 'sampling':
                    if prompt_to_decoder:
                        gen_tokens = model.generate(**encoding,
                                                    decoder_input_ids=encoding_decoder['input_ids'],
                                                    do_sample=True,
                                                    temperature=args.temperature,
                                                    max_length=args.max_len,
                                                    num_return_sequences=args.num_seqs_per_iter,
                                                    decoder_start_token_id=tokenizer.pad_token_id,
                                                    eos_token_id=tokenizer.eos_token_id,
                                                    top_p=0.95)
                    else:
                        gen_tokens = model.generate(**encoding,
                                                    do_sample=True,
                                                    temperature=args.temperature,
                                                    max_length=args.max_len,
                                                    num_return_sequences=args.num_seqs_per_iter,
                                                    eos_token_id=tokenizer.eos_token_id,
                                                    top_p=0.95)

            if gen_tokens is not None:
                if prompt_to_decoder:
                    gen_tokens = gen_tokens[:, encoding_decoder['input_ids'].shape[-1]:]
                gen_seqs = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
            else:
                gen_seqs = None

            if gen_seqs is not None:
                assert len(ids_batch) == 1
                task_id = ids_batch[0]

                for seq_idx, gen_seq in enumerate(gen_seqs):
                    completion_seq = gen_seq
                    for stop_seq in STOP_SEQS:
                        index = completion_seq.find(stop_seq)
                        if index != -1:
                            completion_seq = completion_seq[:index]
                    completion_seq = completion_seq.replace('\t', '    ')
                    all_code = prompt.replace('\t', '    ') + completion_seq

                    completion_seqs.append(
                        {'task_id': task_id,
                         'completion': completion_seq,
                         'all_code': all_code  # final code for evaluation with unit tests
                         }
                    )

        print("Saving results to {}".format(output_file))
        write_jsonl(output_file, completion_seqs)


if __name__ == '__main__':
    main()
