### DACON Competition

# 위성관측 데이터 활용 강수량 산출 경진대회

<img src="md_imgs/header.jpg" alt="md_img" width="100%;" />

## 1. 개요

## 기상 | 위성 이미지 빅데이터와 AI로 NASA보다 정확한 강수량 산출

https://dacon.io/competitions/official/235591/overview/



## 2. Data

해결해야 하는 문제

\-   test.zip 파일 각각 픽셀별 강수량 산출



데이터 설명

\-   GPM(Global Precipitation Measurement) Core 위성의 GMI/DPR 센서에서 북서태평양영역 (육지와 바다를 모두 포함) 에서 관측된 자료

\-   특정 orbit에서 기록된 자료를 40 X 40 형태로 분할(subset) 하여 제공

\-   subset`_`######`_`##.npy 파일로 제공되며, (height, width, channel) 형태

\-   ###### : 위성이 사용되기 시작한 이후로 몇 번째 지구를 돌았는지 나타내는 수(orbit 번호)

\-   ##: 해당 orbit에서 몇 번째 subset인지를 나타내는 수입니다. orbit별로 subset의 개수는 다를 수 있음 (subset 번호)

\-   데이터 출처 및 기타 세부사항은 토론 게시판의 pdf 자료 및 영상 자료 확인

\-   pdf자료: https://dacon.io/competitions/official/235591/talkboard/400589

\-   영상자료: https://dacon.io/competitions/official/235591/talkboard/400598

- 채널 0~8: 밝기 온도 (단위: K, 10.65GHz~89.0GHz)
- 채널 09: 지표 타입 (앞자리 0: Ocean, 앞자리 1: Land, 앞자리 2: Coastal, 앞자리 3: Inland Water)
- 채널 10: GMI 경도: GMI는 마이크로파 이미지 센서 - 약 900km 관측 가능
- 채널 11: GMI 위도
- 채널 12: DPR 경도: DPR은 실제 강수량 예측 레이더 센서 - 다만 관측 폭이 GMI에 비해 작음 (125, 245km 2개의 밴드를 이용하여 관측 중) 따라서 GMI의 이미지 센서를 토대로 나머지를 예측하고 싶어하는 것임.
- 채널 13: DPR 위도
- 채널 14: 강수량 (mm/h, 결측치는 -9999.xxx 형태의 float 값으로 표기)



train.zip

\-   2016~2018 년 관측된 자료 (76,345개)

\-   2016년 자료: orbit 번호 010462 ~ 016152 (25,653개)

\-   2017년 자료: orbit 번호 016154 ~ 021828 (25,197개)

\-   2018년 자료: orbit 번호 021835 ~ 027509 (25,495개)



test.zip 

\-   2019년 관측된 자료 (2,416개)



sample_submission.csv

\-   제출 양식 예시

\-   시각화 참조: https://dacon.io/competitions/official/235591/talkboard/400629



## 3. Evaluation(평가)

```python
# 평가 지표는 MAE를 F1 score로 나눈 값

import numpy as np
from sklearn.metrics import f1_score


def mae_over_fscore(y_true, y_pred):
    '''
    y_true: sample_submission.csv 형태의 실제 값
    y_pred: sample_submission.csv 형태의 예측 값
    '''


    y_true = np.array(y_true)
    y_true = y_true.reshape(1, -1)[0]  
    
    y_pred = np.array(y_pred)
    y_pred = y_pred.reshape(1, -1)[0]
    
    # 실제값이 0.1 이상인 픽셀의 위치 확인
    IsGreaterThanEqualTo_PointOne = y_true >= 0.1
    
    # 실제 값에 결측값이 없는 픽셀의 위치 확인 
    IsNotMissing = y_true >= 0
    
    # mae 계산
    mae = np.mean(np.abs(y_true[IsGreaterThanEqualTo_PointOne] - y_pred[IsGreaterThanEqualTo_PointOne]))
    
    # f1_score 계산 위해, 실제값에 결측값이 없는 픽셀에 대해 1과 0으로 값 변환
    y_true = np.where(y_true[IsNotMissing] >= 0.1, 1, 0)
    
    y_pred = np.where(y_pred[IsNotMissing] >= 0.1, 1, 0)
    
    # f1_score 계산    
    f_score = f1_score(y_true, y_pred) 
    
    # f1_score가 0일 나올 경우를 대비하여 소량의 값 (1e-07) 추가 
    return mae / (f_score + 1e-07) 
```





## 4. Preprocessing (데이터 전처리)





## 5. Result (결과)



## 6. 새로 배운 개념

- tfa.optimizers.SWA
- FPN(Feature Pyramid Networks)