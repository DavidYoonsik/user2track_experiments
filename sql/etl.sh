#!/bin/bash

set -e
CUR_PATH="$(cd "$(dirname "$0")" && pwd)"

if [[ -z $1 ]]; then
    echo 'insert start of yyyymmdd'
    echo "e.g., ./${0} 20191201"
    exit -1
fi

if [[ -z $2 ]]; then
    echo 'insert end of yyyymmdd'
    echo "e.g., ./${0} 20191280"
    exit -1
fi

if [[ -z $3 ]]; then
    echo 'insert padding size'
    echo "e.g., ./${0} 100"
    exit -1
fi

if [[ -z $4 ]]; then
    echo 'insert unique listen threshold'
    echo "e.g., ./${0} 9"
    exit -1
fi

JOB_NAME=ETL
TASK_NAME=User2Track

# destination
OUTPUT_BUCKET=${OUTPUT_BUCKET:-"flo-tmp"}
OUTPUT_DATABASE=${OUTPUT_DATABASE:-"flo_tmp"}
OUTPUT_TRAIN_TABLE=${OUTPUT_TRAIN_TABLE:-"tmp_train"}
OUTPUT_INFER_TABLE=${OUTPUT_INFER_TABLE:-"tmp_infer"}
OUTPUT_GT_TABLE=${OUTPUT_GT_TABLE:-"tmp_gt"}
OUTPUT_META_TABLE=${OUTPUT_META_TABLE:-"tmp_meta"}

${CUR_PATH}/../_spark-sql.sh \
    ${JOB_NAME}-${TASK_NAME} \
    ${CUR_PATH}/../sql/etl.sql \
    OUTPUT_BUCKET=${OUTPUT_BUCKET} \
    OUTPUT_DATABASE=${OUTPUT_DATABASE} \
    OUTPUT_TRAIN_TABLE=${OUTPUT_TRAIN_TABLE} \
    OUTPUT_INFER_TABLE=${OUTPUT_INFER_TABLE} \
    OUTPUT_GT_TABLE=${OUTPUT_GT_TABLE} \
    OUTPUT_META_TABLE=${OUTPUT_META_TABLE} \
    ST=${1} \
    ET=${2} \
    PAD_SIZE=${3} \
    UNIQ_LISTEN=${4} \
    UNIQ_LISTEN_Y=${5}