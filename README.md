# CF USER2TRACK README

## data
- train.npz (train, test, inference 3분류 데이터셋 저장)

## src
- dataset.py (spark-sql)
- func_utils.py, layer_utils.py
- base_layer.py, conv1d_layer.py, conv2d_layer.py
- train.py (학습 & 추론)
- s3_to_memcached.py (학습 및 추론 결과물 S3 업로드 / 개별적 업로드 기능)

## res
- config.dev.yaml, config.real.yaml
- install-python.sh

## sql
- etl.sh
- etl.sql

## flow
- 학습을 위한 데이터 생성
- S3 업로드
- 학습 단계에서 S3 데이터셋 로드
- 학습 진행
- 결과물(체크포인트, 모델 및 메타 데이터) S3 업로드
- (옵션) tb_output/ 경로로 Tensorboard 실행 후, 학습 히스토리 및 모델 그래프 확인
- 추론 진행
- 결과물(.npy) S3 업로드
- DEMO 페이지 결과 비교(이전 모델) 및 확인

## run(PROD=0 for TEST, PROD=1 for REAL)
```
- ./dockerize.sh
- docker run 실행과 함께 AWS 키와 비밀키 값을 넘겨주여야 AWS Bucket 접근 가능
- docker run --runtime=nvidia -it --net host -e AWS_ACCESS_KEY_ID=*** -e AWS_SECRET_ACCESS_KEY=*** 
  DOCEKR_IMAGE_MADE_ON_ABOVE:TAG /bin/bash
- PROD=1 ./runner etl 20191101 20191130 100 9
- 개발서버에서 etl 관련 job을 실행시키고,
- PROD=1 ./runner train 20191131
- 학습 및 모델 확인 후, 추론 단계로 넘어가며 결과물 업로드 및 메트릭 정보 이상유무 확인
- 추론 결과 검증 이후, 최종적으로 나오는 메트릭 정보와 *.npy 파일 추출
```