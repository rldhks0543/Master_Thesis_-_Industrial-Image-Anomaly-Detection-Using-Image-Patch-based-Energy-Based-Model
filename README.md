# (학위논문) 산업 이미지에서의 이상 탐지를 위한 이미지 패치 기반 에너지 모델의 활용

**초록**  
 본 연구에서는 산업 이미지 데이터에서의 이상 감지(Anomaly detection)를 위한 이미지 패치 기반 Energy based model(EBM)을 제시한다. 비지도 학습 방식을 사용하는 EBM은 학습된 데이터와 유사한 분포를 가진 데이터에 대해서는 낮은 값을 반환하고 다른 분포를 가진 데이터에 대해서는 높은 값을 반환하는 특성을 가진다. 이러한 EBM의 특성은 정상 데이터와 다른 분포를 가진 이상 데이터에 대해 높은 값을 도출하여 이상 감지가 가능하게 한다. 그러나 산업 데이터에서 기존의 에너지 모델은 이미지 전체 수준(image-level)에서 학습되어 이미지의 전반적인 외형 변형은 잘 감지하였으나 세부적인 특징을 파악하지 못하였다. 따라서 본 연구에서는 패치 수준(patch-level)에서 학습한 모델과 이미지 전체 수준에서 학습한 모델을 결합하여 전반적인 외형 변형을 식별하면서도 세부적인 변형도 잘 감지할 수 있는 모델을 제안한다. 뿐만 아니라 기존의 EBM과 비교하여 샘플링 알고리즘을 2.4배 빠르게 개선하고 EBM에 배치 정규화(batch norm)를 적용할 수 있는 [JEM++ 모델](https://arxiv.org/pdf/2109.09032.pdf)을 백본 모델로 사용한다. 실험 결과, 기존의 에너지 모델에 비해 AUROC(the Area Under the ROC Curve)의 평균값이 18.3%p 향상되었으며 비교군인 생성 모델들보다 간단한 모델 구조로 최고 성능을 달성할 수 있었다.
***
## 모델의 구조

본 연구에서는 **1. 이미지 패치 기반 에너지 모델(Image Patch-level EBM)** 과 **2. 이미지 전체 기반 에너지 모델(Image-level EBM)** 을 **3. 앙상블한 모델** 을 제시한다.

### 1. 이미지 패치 기반 에너지 모델(Image Patch-level EBM)의 구조

#### 1-1. 학습 구조(Training Process)  
이미지 패치 기반 에너지 모델의 학습 구조는 그림1과 같다. 각 반복(iteration)에 정상 이미지에서 이미지 가운데를 기준으로 가우시안 분포로 랜덤한 패치를 떼어낸다. 이후 이미지 패치와 PYLD-M-N 알고리즘으로 생성된 이미지를 에너지 신경망(Energy CNN)에 넣은 뒤, 각각의 에너지 점수를 뽑아내게 된다. 이 두 에너지 점수를 손실 함수에 넣어 에너지 신경망을 업데이트하게 된다.  
<img width="650" alt="training process" src="https://github.com/rldhks0543/Patch-based_EBM/assets/114603826/63009be0-cfec-4c53-af53-20823f710391">

#### 1-2. 평가 구조(Testing Process)<br/>
실험 이미지(test image) 한 장인 가 존재할 때, n*n의 격자로 이미지를 자른다. 모든 grid에 대하여 에너지 함수를 적용한 뒤, 도출된 에너지 점수들 중 가장 큰 값을 의 최종 이상 점수로 지정한다.  
<img width="650" alt="Testing process" src="https://github.com/rldhks0543/Patch-based_EBM/assets/114603826/8a4b2b04-eaf7-4a67-a844-0f202f98a671">

### 2. 이미지 기반 에너지 모델(Image-level EBM)

이미지 기반 에너지 모델은 일반적인 EBM의 형태이다. patch와 같이 일부 이미지가 아닌 이미지 전체로 EBM이 학습되며, 한장의 전체 이미지에서 한개의 최종 이상 점수를 도출한다.

### 3. 앙상블한 모델

최종적인 이상 점수를 도출하기 위하여, 패치 수준(patch-level)에서 학습한 EBM과 이미지 전체 수준(image-level)으로 학습하는 EBM에서 도출된 이상 점수를 앙상블하여 최종점수로 선정하였다.  
<img width="650" alt="Ensemble process" src="https://github.com/rldhks0543/Patch-based_EBM/assets/114603826/cedd5aa3-e841-43a1-b014-f4f16526e41f">
***
## 실험 결과
본 모델을 적용한 데이터 셋은 제조 산업 이미지 데이터인 [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) 데이터셋이다.

본 논문에서 제안하는 앙상블 된 최종 모델과 비교군 모델들의 성능을 비교하기 위해 표3을 제시한다. 비교군은 EBM과 같이 재구성(reconstruct) 방식의 이상 감지 모델으로 설정하였다. [Baseline EBM](https://arxiv.org/pdf/2105.03270.pdf)의 경우 본 논문에서 제안하는 모델과 달리 8개 층의 기본적인 CNN 활용, PYLD-M-N 샘플링 알고리즘이 아닌 SGLD 샘플링 알고리즘 적용, Batch norm 미적용, 샘플링 과정에서 생성되는 잡음(noise) 벡터 생성 방식에 있어서 차이가 난다. 이에 따라 Baseline EBM이 매우 기본적인 형태임에 성능이 낮을 수 있으나, 발전된 알고리즘들이 적용된 본 논문의 EBM과 비교하기 위하여 비교군에 추가하였다.  
<img width="559" alt="성능표" src="https://github.com/rldhks0543/Patch-based_EBM/assets/114603826/4756cb08-f7fa-4681-8239-7f09beea60f7">
