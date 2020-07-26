#!/bin/bash -l
set -e

CUR_PATH="$(cd "$(dirname "$0")" && pwd)"
export JOB_HOME=${JOB_HOME:-${CUR_PATH}}
echo ">> JOB_HOME: ${JOB_HOME}"

# get PROD env
PROD=${PROD:-0}

# debugging
if [ $PROD -eq 1 ]; then
  CONFIG_FILE=res/config.real.yaml
else
  CONFIG_FILE=res/config.dev.yaml
fi

# set arguments
TASK_NAME=$1
TASK_ARGS_LEN=${#}-1
TASK_ARGS=${@:2:${TASK_ARGS_LEN}}

# execute
case ${TASK_NAME} in
  etl)
    bash ${JOB_HOME}/sql/etl.sh ${TASK_ARGS}
    ;;
  remote_train)
    YYYYMMDD=${TASK_ARGS[0]}
    bash ${JOB_HOME}/remote-runner.sh ${YYYYMMDD}
    ;;
  train)
    YYYYMMDD=${TASK_ARGS[0]}
    python3 scr/u2t_train_to_inference.py --config $CONFIG_FILE --yyyymmdd $YYYYMMDD
    ;;
  upload)
    YYYYMMDD=${TASK_ARGS[0]}
    python3 src/u2t_metric_to_mc.py --config $CONFIG_FILE --yyyymmdd $YYYYMMDD
    ;;
  *)
    echo ">> Wrong task_name \`${TASK_NAME}\`."
    exit 1
    ;;
esac