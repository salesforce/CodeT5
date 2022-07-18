#!/bin/bash
python run_gen.py --do_train --do_eval --do_eval_bleu --do_test  \
	--task summarize --sub_task python --model_type codet5 --data_num -1 \
	--num_train_epochs 3 --warmup_steps 1000 --learning_rate 5e-5 --patience 3 \
	--tokenizer_name=Salesforce/codet5-base  \
	--model_name_or_path /home/aumahesh/w266-summer-2022-project/data/model_mrv_cds_best \
	--data_dir /home/aumahesh/w266-summer-2022-project/data/code-docstring-corpus/data  \
	--cache_path /home/aumahesh/w266-summer-2022-project/data/finetuned/codet5_mrv_cds/cache  \
	--output_dir /home/aumahesh/w266-summer-2022-project/data/finetuned/codet5_mrv_cds/output  \
	--summary_dir /home/aumahesh/w266-summer-2022-project/data/finetuned/codet5_mrv_cds/summary \
	--save_last_checkpoints --always_save_model \
	--res_dir /home/aumahesh/w266-summer-2022-project/data/finetuned/codet5_mrv_cds/results --res_fn mrv_cds_res.txt \
	--train_batch_size 16 --eval_batch_size 8 --max_source_length 256 --max_target_length 128 \
	--save_pretrained \
	2>&1 | tee /home/aumahesh/w266-summer-2022-project/data/finetuned/codet5_mrv_cds/log/log.txt

