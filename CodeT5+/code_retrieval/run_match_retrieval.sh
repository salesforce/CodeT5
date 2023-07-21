export TOKENIZERS_PARALLELISM=false
# choices: ruby javascript go python java php AdvTest cosqa
LANG=ruby
BS=256
CODE_LEN=360
TEXT_LEN=64
TOPK=32
MODEL_NAME=Salesforce/codet5p-220m-bimodal
DATA_DIR=/path/to/data

TRG_DIR=saved_models/${LANG}/codet5p_220m_bimodal_TL${TEXT_LEN}_CL${CODE_LEN}_top${TOPK}
mkdir -p $TRG_DIR
echo 'Target dir: '$TRG_DIR

python eval_match_retrieval.py --model_name $MODEL_NAME --lang $LANG --output_dir $TRG_DIR \
  --data_dir $DATA_DIR --max_text_len $TEXT_LEN --max_code_len $CODE_LEN --batch_size $BS --top_k $TOPK \
  2>&1 | tee ${TRG_DIR}/log.txt
