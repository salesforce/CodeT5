#!/bin/bash
set -euox pipefail

# ENV vars
# model
# input_path
# temp
# top_p
# max_len
# n_samples
# gpu_num

model=${model:-"instructcodet5p-16b"}
input_path=${input_path:-""}

temp=${temp:-0.8}
top_p=${top_p:-0.95}
max_len=${max_len:-512}
pred_num=${n_samples:-200}
num_seqs_per_iter=2 # 25 for 350M and 770M, 10 for 2B, 8 for 6B, 2 for 16B on A100-40G
gpu_num=${gpu_num:-1}

output_path=preds/${model}_T${temp}_P${top_p}_N${pred_num}

mkdir -p "${output_path}"
echo "Output path: $output_path"
echo "Model to eval: $model"

# 164 problems, 21 per GPU if GPU=8
batch_size=164
index=0
for ((i = 0; i < $gpu_num; i++)); do
  start_index=$((i * $batch_size))
  end_index=$(((i + 1) * $batch_size))

  gpu=$((i))
  echo 'Running process #' ${i} 'from' $start_index 'to' $end_index 'on GPU' ${gpu}
  CMD="CUDA_VISIBLE_DEVICES=$gpu python generate_codet5p.py --model Salesforce/${model} \
      --start_index ${start_index} --end_index ${end_index} \
      --temperature ${temp} \
      --num_seqs_per_iter ${num_seqs_per_iter} \
      --N ${pred_num} \
      --max_len ${max_len} \
      --output_path ${output_path}"
  ((index++))
  (
    if [ -z "$input_path" ]
    then
      eval "$CMD"
    else
      eval "$CMD" --input_path "${input_path}"
    fi
  ) &
  if (($index % $gpu_num == 0)); then wait; fi
done
