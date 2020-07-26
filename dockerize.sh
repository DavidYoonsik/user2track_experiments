#!/bin/bash
set -e
DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $DIR
echo ">> DIR: ${DIR}"

UPLOAD=${UPLOAD:-0}
PROD=${PROD:-0}

DOCKERFILE=$DIR/Dockerfile
APPNAME=flo-reco/workflow

if [ $PROD -eq 1 ]; then
  BASEIMGTAG=base-current
  APPTAG=user2track-current
else
  BASEIMGTAG=base-dev
  APPTAG=user2track-test
fi

BASEIMG="865306278000.dkr.ecr.ap-northeast-2.amazonaws.com/flo-reco/workflow:$BASEIMGTAG"
IMGNAME="865306278000.dkr.ecr.ap-northeast-2.amazonaws.com/$APPNAME:$APPTAG"

echo "--- Docker build arguments ---"
echo "    DOCKERFILE: $DOCKERFILE"
echo "    IMGNAME: $IMGNAME"
echo "    BASEIMG: $BASEIMG"
echo "------------------------------"
docker build \
    --tag ${IMGNAME} \
    --build-arg BASEIMG=${BASEIMG} \
    -f $DOCKERFILE \
    --force-rm \
    $DIR

if [ $UPLOAD -eq 1 ]; then
    echo '>> create ecr'
    AWSCLI_VERSION=$(aws --version | awk '{ split($1, version, "/"); print version[2] }')
    if [[ "$AWSCLI_VERSION" < "2.0" ]]; then
        eval $(aws ecr get-login --no-include-email)
    else
        aws ecr get-login-password \
            | docker login --username AWS --password-stdin $IMGNAME
    fi
    EXISTS=$(aws ecr describe-repositories --output=text | grep $APPNAME) || echo -n
    if [[ -z $EXISTS ]]; then
        aws ecr create-repository --repository-name $APPNAME
    fi

    echo '>> docker push'
    docker push $IMGNAME
fi