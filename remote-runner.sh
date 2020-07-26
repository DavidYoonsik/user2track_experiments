#!/bin/bash -l
set -e

YYYYMMDD=$1
PROD=${PROD:-0}

SECRET_INFO=$(aws secretsmanager get-secret-value --secret-id user2track/access | jq .SecretString | jq fromjson)

AWS_ACCESS_KEY_ID=$(echo $SECRET_INFO | jq -r .aak)
AWS_SECRET_ACCESS_KEY=$(echo $SECRET_INFO | jq -r .asak)
AWS_DEFAULT_REGION="ap-northeast-2"
AWS_DEFAULT_OUTPUT="json"

MCP_HOST=$(echo $SECRET_INFO | jq -r .mcp_host)
MCP_PORT=$(echo $SECRET_INFO | jq -r .mcp_port)
MCP_USER=$(echo $SECRET_INFO | jq -r .mcp_user)
MCP_PWD=$(echo $SECRET_INFO | jq -r .mcp_pwd)

DOCKER_IMAGE_NAME="865306278000.dkr.ecr.ap-northeast-2.amazonaws.com/flo-reco/workflow"

if [ ${PROD} -eq 1 ]; then
  DOCKER_IMAGE_TAG="user2track-current"
else
  DOCKER_IMAGE_TAG="user2track-test"
fi

AWSCLI_VERSION=$(aws --version | awk '{ split($1, version, "/"); print version[2] }')

# --runtime=nvidia param should be passed when run train only(GPU MODE)
DOCKER_OPTS="--rm -i --runtime=nvidia --net host -e PROD=$PROD -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY -e AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION \
  -e AWS_DEFAULT_OUTPUT=$AWS_DEFAULT_OUTPUT" -v "/data01":"/data01"

if [[ "$AWSCLI_VERSION" < "2.0" ]]; then
  sshpass -p $MCP_PWD ssh -o StrictHostKeyChecking=no $MCP_USER@$MCP_HOST -p $MCP_PORT """
  eval \$(aws --region ap-northeast-2 ecr get-login --no-include-email)
  docker run $DOCKER_OPTS \
  $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG \
  /opt/app/runner.sh train ${YYYYMMDD}
  """
else
  sshpass -p $MCP_PWD ssh -o StrictHostKeyChecking=no $MCP_USER@$MCP_HOST -p $MCP_PORT """
  aws ecr get-login-password | docker login --username AWS --password-stdin $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG
  docker run $DOCKER_OPTS \
  $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG \
  /opt/app/runner.sh train ${YYYYMMDD}
  """
fi
