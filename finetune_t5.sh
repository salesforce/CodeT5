#!/bin/bash
python run_gen.py --do_train --do_eval --do_eval_bleu --do_test  \
	--task summarize --sub_task python --model_type codet5 --data_num -1 \
	--num_train_epochs 3 --warmup_steps 1000 --learning_rate 5e-5 --patience 3 \
	--tokenizer_name=Salesforce/codet5-base  \
	--model_name_or_path Salesforce/codet5-base-multi-sum \
	--data_dir /home/aumahesh/w266-summer-2022-project/data/code-docstring-corpus/data  \
	--cache_path /home/aumahesh/w266-summer-2022-project/data/finetuned/codet5/cache  \
	--output_dir /home/aumahesh/w266-summer-2022-project/data/finetuned/codet5/output  \
	--summary_dir /home/aumahesh/w266-summer-2022-project/data/finetuned/codet5/summary \
	--save_last_checkpoints --always_save_model \
	--res_dir /home/aumahesh/w266-summer-2022-project/data/finetuned/codet5/results --res_fn t5_res.txt \
	--train_batch_size 16 --eval_batch_size 8 --max_source_length 256 --max_target_length 128 \
	--save_pretrained \
	2>&1 | tee /home/aumahesh/w266-summer-2022-project/data/finetuned/codet5/log/log.txt

