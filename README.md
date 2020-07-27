# CF USER2TRACK README

## data
- 학습 데이터, 테스트 데이터, 추론 데이터, 메타 데이터, Track2Track 데이터, G.T 데이터
- 업로드에 필요한 데이터(*.npy, candidates.gz, etc,.)

## src
- 데이터 처리, 학습, 추론 및 결과물 업로드(s3) 과정을 End-to-End 구성
- metric 수치 결과 upload(s3 --> memCache)

## res
- config.dev.yaml, config.prod.yaml
- install-python.sh

## sql
- etl.sh
- etl.sql

## *.npy, candidates.gz upload
- flo-tmp/model/user2track/latest/*.npy
- flo-tmp/model/user2track/latest/candidates.gz

## external volume
- MCP-GPU01 서버 내에서 디스크 공간 이슈를 해결하기 위해서
- /data01 디렉토리를 상대적으로 용량이 큰 데이터를 저장하는 경로로 사용

## Tensorboard 경로
- MCP-GPU01 서버
- http://172.21.74.111:9006

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