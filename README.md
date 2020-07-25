## ConvLSTM 기반 지자체별 코로나바이러스 예측
> 이미지적 접근을 통한 국토의 공간적 특성 반영한 코로나 확산 예측 

- 제3회 정부혁신 끝장개발대회 메이커톤, 아이디어톤.
- by 승찬용 (seung-kyo Jin, chan-hee You, ho-yong Kim) <br>
프로젝트 소개 페이지: <https://civichack.or.kr/?page=maker&mode=view&pno=14><br>
---
![handmade_extended_color_20](https://user-images.githubusercontent.com/30429632/88447006-e3629c80-ce69-11ea-9f66-3fc666da54c0.gif) <br>
그림1. 지역별 확진자 수 추이 (초당 20장)


#### 문제 인식 
코로나-19를 초기대책을 마련하는데는 과정에서 가용할 수 있는 인적, 물적자원이 한정적

#### 제안 이유
- 지역별 확진자 발생 추이를 예측함으로써 중앙정부 차원에서 코로나 2차 대유행이나 장기화나 코로나 종결등을 사전에 대비할 수 있도록 지원
-  시각적으로 코로나 심각도의 트랜드를 직관적으로 파악

#### 출품작의 역할
  - 코로나 상황 종료, 지역별 확진자가 급증하는 상황같은 특이 경우를 예측하여 정부의 정책 대응을 지원
  - 개발한 프로토타입 모델은 우리나라의 세분화된 지자체 데이터나 전 세계 데이터를 대상으로 모델을 적용해 확장
  
#### 방법
<p>
순환신경망 계열 모델 중 하나인 convolutional lstm(convlstm)은 모델의 장기기억 문제를 보완한 LSTM모델과 데이터의 지역적인 특징을 포착하는데 강점이 있는 CNN 모델의 장점을 더하였다. 우리는 데이터셋의 크기를 더하기 위해 대회에서 제공하는 데이터셋에 코로나바이러스감염증-19에서 추출한 데이터를 더하고 도시별, 시간대별로 구분해 학습용 데이터셋을 구축했다. 데이터를 이미지로 변환하고 convlstm을 학습시켜 미래 시점의 확진자 추이를 예측하였다.
</p>
  
#### 출품작의 제작 과정
1. 문제 인식: 기존의 수치적 접근은 점염병의 특성상 공간적 특성을 제대로 반영할 수 없음
2. 데이터셋 확보: 지역별 신규확진자, 격리자 데이터를 확보
3. 데이터 변환: 데이터를 대한민국의 지도에 플로팅을 하여 이미지화하였다. 
4. 모델 학습: 공간적 특성을 반영하고 있으므로 인공지능으로 학습시켜 예측

#### 결과
![pred0](https://user-images.githubusercontent.com/30429632/88448392-815d6380-ce78-11ea-8b38-28216d7faf5e.png) <br>
그림2. 예측 이미지 <br>

![real0](https://user-images.githubusercontent.com/30429632/88448414-d13c2a80-ce78-11ea-807e-800ffc46155e.png)
그림3. 결과 이미지 <br>

#### 프로토타입 모델의 확장
광역자치단체보다 세분화된 지자체나 전 세계를 대상으로하는 데이터를 수집해 예측의 범위를 넓히고 정확도 향상 

#### 활용 데이터
<https://github.com/jihoo-kim/Data-Science-for-COVID-19> <br>
코로나바이러스감염증-19(중앙방역대책본부)
