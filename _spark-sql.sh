#!/bin/bash
set -e
DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $DIR

if [ $# -lt 2 ]; then
cat << EOF
>> Usage: $0 <job name> <sql absolute path> [hivevars]
Required arguments:
    <job name>: spark job name (e.g., User2TrackETL)
    <sql absolute path>: the absolute path of a SQL file
Optional arguments:
    [hivevars]: hive variables (e.g., OUTPUT_BUCKET=flo-tmp OUTPUT_DATABASE=flo_tmp)
EOF
    exit 1
fi

if [ ! -f $2 ]; then
    echo "$2 does not exist"
    exit 1
fi

# set arguments
SPARK_JOB_NAME=$1
SPARK_SQL_PATH=$2
echo ">> SPARK_JOB_NAME: $SPARK_JOB_NAME"
echo ">> SPARK_SQL_PATH: $SPARK_SQL_PATH"
SPARK_HIVEVARS_LEN=${#}-2
SPARK_HIVEVARS=""
for hivevar in "${@:3:${SPARK_HIVEVARS_LEN}}"; do
    SPARK_HIVEVARS="$SPARK_HIVEVARS --hivevar $hivevar"
    echo ">>> $hivevar"
done
echo ">> SPARK_HIVEVARS: $SPARK_HIVEVARS"
echo

# execute spark-sql
spark-sql \
    --master yarn \
    --name $SPARK_JOB_NAME \
    \
    --conf spark.driver.port=60001 \
    --conf spark.driver.blockManager.port=60002 \
    --conf spark.driver.extraJavaOptions='-XX:MaxPermSize=1024m -XX:PermSize=256m -Dfile.encoding=utf-8' \
    \
    --conf spark.executor.cores=1 \
    --conf spark.executor.memory=4g \
    --conf spark.driver.memory=4g \
    --conf spark.yarn.maxAppAttempts=1 \
    \
    $SPARK_HIVEVARS \
    \
    -f ${SPARK_SQL_PATH}