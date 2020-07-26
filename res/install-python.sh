#!/bin/bash

set -e

PYTHON_VERSION=3.6.8

CONFIGURE_OPTS=--enable-shared pyenv install ${PYTHON_VERSION}
pyenv global ${PYTHON_VERSION}

mkdir -p $HOME/.config/pip
cat <<EOF >$HOME/.config/pip/pip.conf
[global]
extra-index-url = http://flo-pypi.s3-website.ap-northeast-2.amazonaws.com
trusted-host = flo-pypi.s3-website.ap-northeast-2.amazonaws.com
EOF