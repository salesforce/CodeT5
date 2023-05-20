from human_eval.data import read_problems, write_jsonl, stream_jsonl
import glob 
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()

# Inputs
parser.add_argument(
    '--path',
    type=str,
    help="")
parser.add_argument(
    '--out_path',
    type=str,
    help="")
parser.add_argument(
    '--add_prompt',
    action='store_true',
    help='')

args = parser.parse_args()


files = sorted(glob.glob(args.path + '/*.jsonl'))
print("{} files in {}".format(len(files), args.path))

problems = read_problems('data/HumanEval.jsonl.gz')

output = []
for code_file in tqdm(files, total=len(files)):
    codes = [c for c in stream_jsonl(code_file)]
    if args.add_prompt: 
        for code in codes: 
            task_id = code['task_id']
            prompt = problems[task_id]['prompt'] 
            if 'def' in code['completion']: 
                def_line = code['completion'].index('def')
                completion = code['completion'][def_line:]
                next_line = completion.index('\n')
                completion = code['completion'][def_line+next_line+1:]
                code['all_code'] = prompt + completion 
    
    output += codes 
    
print("save to {}".format(args.out_path))
write_jsonl(args.out_path, output)