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

본 논문에서 제안하는 앙상블 된 최종 모델과 비교군 모델들의 성능을 비교하기 위해 표3을 제시한다. 비교군은 EBM과 같이 재구성(reconstruct) 방식의 이상 감지 모델으로 설정하였다. [Baseline EBM[7]](https://arxiv.org/pdf/2105.03270.pdf)의 경우 이미지 기반 에너지 모델(Image-level EBM)과 같이 이미지 전체를 학습하는 방식은 같으나, 본 논문에서 제안하는 모델과 달리 8개 층의 기본적인 CNN 활용, PYLD-M-N 샘플링 알고리즘이 아닌 SGLD 샘플링 알고리즘 적용, Batch norm 미적용, 샘플링 과정에서 생성되는 잡음(noise) 벡터 생성 방식에 있어서 차이가 난다. 이에 따라 Baseline EBM이 매우 기본적인 형태임에 성능이 낮을 수 있으나, 발전된 알고리즘들이 적용된 본 논문의 EBM과 비교하기 위하여 비교군에 추가하였다.  
  
우선 성능 비교표(표3)에서 Object 카테고리는 bottle, Hazelnut, Toothbrush를 제외하고 비교군에 비해 최고 성능을 보였으며, 특히 Texture 카테고리의 경우는 평균 점수가 0.958점으로 최고 성능을 보였다. 전체 카테고리 평균 성능 또한 비교군에 비해 최고 성능을 보였으며, Baseline EBM에 비해 18.3%p 차이를 보이며 성능이 크게 차이남을 알 수 있었다.  
  
<img width="559" alt="성능표" src="https://github.com/rldhks0543/Patch-based_EBM/assets/114603826/4756cb08-f7fa-4681-8239-7f09beea60f7">
  
***
## Appendix

**A. 모델 세부 설정**  
  
<img width="253" alt="appendix A" src="https://github.com/rldhks0543/Patch-based_EBM/assets/114603826/c7875276-b78e-4e15-8cb8-f2d968430b4d">
  
  
**B. PYLD-M-N 알고리즘**  
  
PYLD-M-N으로 생성된 샘플은 그림4와 같다. Image-level EBM은 디테일한 텍스쳐는 뭉개지지만 이미지의 전체적인 외형을 재구성(reconstruct)한 샘플이 생성 됨을 확인할 수 있었다. 반면에 Patch-level EBM은 이미지의 전체적인 외형보다 텍스쳐를 위주로 샘플이 생성됨을 확인할 수 있다.  
<img width="523" alt="appendix B" src="https://github.com/rldhks0543/Patch-based_EBM/assets/114603826/7a90daa3-c03c-4110-9084-7bb444b2be00">
  
  
**C. Grad-CAM 시각화 결과**  
  
모델이 중점적으로 집중하고 있는 부분을 확인하기 위하여 Grad-CAM[18]을 통하여 시각화를 진행하였다. Image-level EBM의 경우 전체적인 외형에 모델이 집중하는 반면에
patch-level EBM의 경우 이미지의 텍스쳐에 대한 변화(그림5에서 페인트 부분)에 더욱 집중하는 것을 확인할 수 있었다.  
<img width="495" alt="appendix C" src="https://github.com/rldhks0543/Patch-based_EBM/assets/114603826/87ce08cd-0cf2-4d02-87b0-0f9e418ac3e5">
  
  
**D. 카테고리별 이상 감지 분류 그래프**  
  
본 모델에서는 에너지 CNN을 통과하였을 때 도출된 단일 값(scalar)으로 이상 감지를 실시한다. 따라서 일정한 임계치를 설정하여 정상, 비정상을 판단하게 된다. 15개 카테고리에 대한 이상 점수와 임계치를 시각화한 히스토그램 그래프는 표7과 같다.
  
<img width="488" alt="appendix D" src="https://github.com/rldhks0543/Patch-based_EBM/assets/114603826/e51714b8-bf86-4781-9091-c0d0cdfb4ecf">

***
## Abstract
  
1. Samet Akcay, Amir Atapour-Abarghouei, and Toby P. Breckon. "Ganomaly: Semi-supervised anomaly detection via adversarial training." in ACCV, vol. 3, no. 14, pp. 622-637,   2018.

2. Samet Akcay, Amir Atapour-Abarghouei, and Toby P. Breckon. "Skip-ganomaly: Skip connected and adversarially trained encoder-decoder anomaly detection." in IEEE/IJCNN, pp. 1-8, 2019.

3. Paul Bergmann, et al. “Mvtec ad–acomprehensive real-world dataset for unsupervised anomaly detection.” in IEEE /CVF CVPR, pp. 9592-9600,  2019.

4. Niv Cohen, Yedid Hoshen. "Sub-image anomaly detection with deep pyramid correspondences." in arXiv preprint arXiv, 2020.

5. Thomas Defard, et al. "Padim: a patch distribution modeling framework for anomaly detection and localization." in ICPR, vol. 12664, pp. 475-489, 2021.

6. Yilun Du and Igor Mordatch. "Implicit generation and generalization in energy-based models." in NeurIPS, 2019.

7. Ergin Utku Genc, et al. " Energy-based anomaly detection and localization." in Energy Based Models Workshop - ICLR, 2021.

8. Ian Goodfellow, et al. "Generative adversarial nets." in NeurIPS, vol. 27, 2014.

9. Will Grathwohl, et al. "Your classifier is secretly an energy based model and you should treat it like one." in ICLR, 2020.

10. Geoffrey E Hinton. "Training products of experts by minimizing contrastive divergence." in Neural computation, 2002.

11. Diederik Kingma and Jimmy Ba.  "Adam: A method for stochastic optimization." in ICLR, 2015.

12. Diederik Kingma and Max Welling. "Auto-encoding variational bayes."  in ICLR, 2014.

13. Yann LeCun, et al. "A tutorial on energy-based learning." in Predicting structured data, 2006.

14. Erik Nijkamp, Song-Chun Zhu, and Ying Nian Wu. "Learning non-convergent non-persistent short-run MCMC toward Energy-based model." in NeurIPS, vol. 32,   2019.

15. Oliver Rippel, Patrick Mertens, and Dorit Merhof. "Modeling the distribution of normal data in pre-trained deep features for anomaly detection." in IEEE/CVF CVPR, pp.6726-6733, 2021.

16. Karsten Roth, et al. "Towards total recall in industrial anomaly detection." in IEEE/CVF CVPR, pp.14318-14328, 2022.

17. Thomas Schlegl, et al. "Unsupervised anomaly detection with generative adversarial networks to guide marker discovery." in IPMI, pp.146-157, 2017.

18. Ramprasaath R Selvaraju, et al. "Grad-cam: Visual explanations from deep networks via gradient-based localization." in  IEEE/ICCV, pp. 618-626, 2017.

19. Kihyuk Sohn, et al. "Self-Supervised Learning for Anomaly Detection and Localization." in U.S. Patent Application, vol. 17,  2022.

20. Yang Song and Diederik P. Kingma. "How to train your energy-based models." in arXiv preprint arXiv, 2021.

21. Ta-wei Tang, et al. "Anomaly detection neural network with dual auto-encoders GAN and its industrial inspection applications." in Sensors, vol. 20, no.12, pp.3336, 2020.
22. Max Welling and Yee W Teh. "Bayesian learning via stochastic gradient langevin dynamics." in ICML, pp. 681-688, 2011.

23. Xiulong Yang and Shihao Ji. "Jem++: Improved techniques for training jem." in IEEE/CVF ICCV, pp. 6494-6503, 2021.

24. Sergey Zagoruyko and Nikos Komodakis. "Wide residual networks." in BMVC, 2016.

25. Dinghuai Zhang, et al. "You only propagate once: Accelerating adversarial training via maximal principle." in NeurIPS, vol. 32, 2019.

26. Jun-Yan Zhu, et al. "Unpaired image-to-image translation using cycle-consistent adversarial networks." in IEEE/ICCV, pp.2223-2232, 2017.
