WORKDIR="path_to_your_dir/CodeT5"
export PYTHONPATH=$WORKDIR

TASK=${1}
SUB_TASK=${2}
MODEL_TAG=${3}
GPU=${4}
DATA_NUM=${5}
BS=${6}
LR=${7}
SRC_LEN=${8}
TRG_LEN=${9}
PATIENCE=${10}
EPOCH=${11}
WARMUP=${12}
MODEL_DIR=${13}
SUMMARY_DIR=${14}
RES_FN=${15}

if [[ $DATA_NUM == -1 ]]; then
  DATA_TAG='all'
else
  DATA_TAG=$DATA_NUM
  EPOCH=1
fi

FULL_MODEL_TAG=${MODEL_TAG}_${DATA_TAG}_lr${LR}_bs${BS}_src${SRC_LEN}_trg${TRG_LEN}_pat${PATIENCE}_e${EPOCH}

if [[ ${SUB_TASK} == none ]]; then
  OUTPUT_DIR=${MODEL_DIR}/${TASK}/${FULL_MODEL_TAG}
else
  OUTPUT_DIR=${MODEL_DIR}/${TASK}/${SUB_TASK}/${FULL_MODEL_TAG}
fi

CACHE_DIR=${OUTPUT_DIR}/cache_data
RES_DIR=${OUTPUT_DIR}/prediction
LOG=${OUTPUT_DIR}/train.log
mkdir -p ${OUTPUT_DIR}
mkdir -p ${CACHE_DIR}
mkdir -p ${RES_DIR}

if [[ $MODEL_TAG == roberta ]]; then
  MODEL_TYPE=roberta
  TOKENIZER=roberta-base
  MODEL_PATH=roberta-base
elif [[ $MODEL_TAG == codebert ]]; then
  MODEL_TYPE=roberta
  TOKENIZER=roberta-base
  MODEL_PATH=microsoft/codebert-base
elif [[ $MODEL_TAG == bart_base ]]; then
  MODEL_TYPE=bart
  TOKENIZER=facebook/bart-base
  MODEL_PATH=facebook/bart-base
elif [[ $MODEL_TAG == codet5_small ]]; then
  MODEL_TYPE=codet5
  TOKENIZER=Salesforce/codet5-small
  MODEL_PATH=Salesforce/codet5-small
elif [[ $MODEL_TAG == codet5_base ]]; then
  MODEL_TYPE=codet5
  TOKENIZER=Salesforce/codet5-base
  MODEL_PATH=Salesforce/codet5-base
fi

if [[ ${TASK} == 'clone' ]]; then
  RUN_FN=${WORKDIR}/run_clone.py
elif [[ ${TASK} == 'defect' ]] && [[ ${MODEL_TYPE} == 'roberta' ||  ${MODEL_TYPE} == 'bart' ]]; then
  RUN_FN=${WORKDIR}/run_defect.py
else
  RUN_FN=${WORKDIR}/run_gen.py
fi


CUDA_VISIBLE_DEVICES=${GPU} \
  python ${RUN_FN}  \
  --do_train --do_eval --do_eval_bleu --do_test  --save_last_checkpoints --always_save_model \
  --task ${TASK} --sub_task ${SUB_TASK} --model_type ${MODEL_TYPE} --data_num ${DATA_NUM}  \
  --num_train_epochs ${EPOCH} --warmup_steps ${WARMUP} --learning_rate ${LR}e-5 --patience ${PATIENCE} \
  --tokenizer_name=${TOKENIZER} --tokenizer_path=${WORKDIR}/tokenizer/salesforce \
  --model_name_or_path=${MODEL_PATH} --output_dir ${OUTPUT_DIR}  --summary_dir ${SUMMARY_DIR} \
  --data_dir ${WORKDIR}/data  --cache_path ${CACHE_DIR} --res_dir ${RES_DIR} --res_fn ${RES_FN} \
  --train_batch_size ${BS} --eval_batch_size ${BS} --max_source_length ${SRC_LEN} --max_target_length ${TRG_LEN} \
  2>&1 | tee ${LOG}
