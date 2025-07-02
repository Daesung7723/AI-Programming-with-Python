

# **PyTorch 및 LLM 기반 AI 서비스 개발 과정: 1주 1일차 \- AI 기초 및 개발 환경 구축**

## **1\. 과정 소개 및 오리엔테이션**

본 보고서는 'PyTorch 및 LLM 기반 AI 서비스 개발 과정'의 1주 1일차 강의 자료를 상세하게 다룹니다. 이 과정은 수강생이 파이토치(PyTorch)와 구글 코랩(Google Colab)을 활용하여 딥러닝 이론, LLM 파인튜닝, 상용 API 활용법, 그리고 AI 서비스 개발의 전 과정을 2주 동안 학습하도록 설계되었습니다.1 궁극적으로 수강생은 상용 AI 소프트웨어 개발 능력을 갖추는 것을 목표로 합니다.1

### **1.1. 과정 개요 및 학습 목표**

본 교육 과정은 총 10일, 80시간으로 구성되어 있으며, 각 주차별로 체계적인 학습 목표를 제시합니다.1 전체 과정의 핵심 학습 목표는 다음과 같습니다:

* 파이토치, 구글 코랩 기반 개발 환경 구축 1  
* DNN, CNN, RNN 모델 구조 이해 및 구현 1  
* Hugging Face 라이브러리 기반 언어 모델 파인튜닝 1  
* 임베딩 및 벡터 데이터베이스 개념 이해와 활용 1  
* 상용 LLM API 활용 AI 서비스 개발 1  
* 팀 프로젝트를 통한 AI 서비스 개발 과정 전체 수행 1

이러한 목표들은 단순한 이론 습득을 넘어, 실제 산업 현장에서 요구되는 실용적이고 문제 해결 중심의 역량 강화에 중점을 두고 있음을 보여줍니다. 이는 이론 강의뿐만 아니라 사례, 실습, 팀 프로젝트 등 실질적인 적용 기회를 통해 학습 효과를 극대화하려는 의도를 반영합니다. 따라서 강의 자료는 이론적 깊이와 함께 실용적 활용성을 동시에 고려하여 구성됩니다.

### **1.2. 1주 1일차 학습 목표 및 내용**

1주차는 '딥러닝 기초 및 핵심 모델'에 중점을 두며 1, 그 첫날인 1일차는 '과정 소개 및 개발 환경'으로 8시간이 배정되어 있습니다.1 1일차의 세부 학습 내용은 다음과 같습니다:

* AI, 머신러닝, 딥러닝 개념 및 역사 1  
* 구글 Colab 사용법 (GPU, 파일 시스템, 명령어) 1  
* Numpy, Pandas, Matplotlib 사용법 실습 1

1일차 학습 내용의 순서는 개념 이해에서부터 개발 환경 구축, 그리고 데이터 처리의 기본기 다지기로 이어지는 매우 논리적인 흐름을 따릅니다. AI, 머신러닝, 딥러닝 개념 이해는 기술의 본질과 필요성을 제공하고, Google Colab은 실제 코드를 실행할 환경과 방법을 제시하며, Numpy, Pandas, Matplotlib은 데이터 처리의 핵심 도구를 소개합니다. 이처럼 각 단계가 다음 단계의 필수적인 선행 지식으로 작용하여, 학습자가 복잡한 AI 개발 과정을 효율적으로 습득할 수 있도록 체계적으로 설계되었습니다. 이는 학습자가 개념 없이 도구만 다루거나, 환경 설정 없이 라이브러리만 배우는 비효율을 방지하고, 각 섹션의 중요성과 연결성을 강조하며 학습 몰입도를 높이는 데 기여합니다.

## **2\. AI, 머신러닝, 딥러닝 개념 및 역사**

AI 기술의 근간을 이루는 핵심 개념들을 명확히 정의하고, 이들이 어떻게 발전해왔는지 역사적 흐름을 통해 이해를 돕습니다.

### **2.1. 이론강의: AI, 머신러닝, 딥러닝의 정의와 관계**

인공지능(AI), 머신러닝(ML), 딥러닝(DL)은 서로 밀접하게 관련되어 있지만, 각기 다른 범주와 특성을 지닌 개념입니다. 이들의 관계는 머신러닝이 인공지능의 하위 집합이고, 딥러닝은 다시 머신러닝의 하위 집합이라는 계층적 구조로 설명할 수 있습니다.2 즉, 모든 딥러닝은 머신러닝에 속하고, 모든 머신러닝은 인공지능에 속하지만, 모든 인공지능이 머신러닝인 것은 아닙니다.2

**인공지능(AI)의 정의**

인공지능은 일반적으로 인간의 지능이 필요하거나 인간이 분석할 수 있는 것보다 규모가 큰 데이터를 포함하는 방식으로 추론, 학습 및 행동할 수 있는 컴퓨터 및 기계를 구축하는 과학 분야입니다.2 AI는 컴퓨터 공학, 데이터 및 분석, 소프트웨어 엔지니어링, 심지어 철학을 비롯한 많은 분야를 아우르는 광범위한 분야입니다.2 비즈니스 수준에서 AI는 데이터 분석, 예상 및 예측, 자연어 처리, 추천, 머신 자동화, 지능형 데이터 검색 등 다양한 사용 사례를 포함하는 일련의 기술을 포괄합니다.2

**머신러닝(ML)의 정의**

머신러닝은 인공지능의 하위 집합으로, 명시적인 프로그래밍 없이도 시스템이 자율적으로 학습하고 개선할 수 있도록 지원합니다.2 주로 훈련 데이터를 통해 학습된 속성을 기반으로 예측하는 것에 초점을 둡니다.4 머신러닝 알고리즘은 패턴과 데이터를 인식하고 시스템에 새로운 데이터가 입력될 때 예측을 실행하는 방식으로 작동합니다.2

머신러닝은 데이터를 학습하는 방식에 따라 크게 세 가지 주요 패러다임으로 나뉩니다:

* **지도 학습 (Supervised Learning):** 라벨이 지정된 학습 데이터(정형 데이터)를 사용하여 특정 입력을 출력에 매핑하는 머신러닝 모델입니다.2 출력이 알려져 있고(예: 사과 그림 인식) 모델은 알려진 출력 데이터로 학습됩니다.2 일반적인 알고리즘에는 선형 회귀, K-최근접 이웃, 나이브 베이즈, 결정 트리 등이 있습니다.2  
* **비지도 학습 (Unsupervised Learning):** 라벨이 지정되지 않은 데이터(비정형 데이터)를 사용하여 패턴을 학습하는 머신러닝 모델입니다.2 출력을 미리 알 수 없으며, 알고리즘은 사람의 정보 입력 없이 데이터로부터 학습하여 속성을 기반으로 한 그룹으로 분류합니다 (예: 사과와 바나나 사진 스스로 분류).2 비지도 학습은 설명 모델링과 패턴 일치에 우수합니다.2  
* **강화 학습 (Reinforcement Learning):** 일련의 시행착오 실험을 통해 '실습하여 학습'하는 머신러닝 모델입니다.2 '에이전트'는 성능이 원하는 범위 내에 있을 때까지 피드백 루프를 통해 정의된 작업을 수행하는 방법을 학습합니다. 에이전트는 작업을 잘 수행할 때 긍정적인 강화를 받고 제대로 수행하지 않을 때는 부정적인 강화를 받습니다 (예: Google 연구자들이 바둑 게임을 플레이하도록 강화 학습 알고리즘을 학습시킨 경우, 이 모델은 바둑 규칙에 대한 사전 지식이 없었지만 학습을 통해 인간 플레이어를 이길 수 있는 지점까지 발전했습니다).2

**딥러닝(DL)의 정의**

딥러닝은 머신러닝의 하위 집합으로, '신경망'을 통해 인공지능을 만드는 머신러닝의 한 종류입니다.3 인공 신경망을 사용하여 정보를 처리하고 분석하며, 신경망이 3개 이상의 레이어로 구성된 경우 이를 '딥(Deep)'이라고 부르며, 따라서 딥러닝이라고 합니다.2 딥러닝 알고리즘은 인간 뇌의 활동에서 영감을 얻었으며 논리적 구조를 갖춘 데이터를 분석하는 데 사용됩니다.2 이미지 및 음성 인식, 객체 감지, 자연어 처리 등 오늘날 우리가 인공지능이라고 생각하는 많은 작업에 딥러닝이 활용됩니다.2

딥러닝에 사용되는 일반적인 신경망 유형은 다음과 같습니다: 순방향 신경망(FF), 순환 신경망(RNN), 장단기 메모리(LSTM), 컨볼루셔널 신경망(CNN), 생성적 적대 신경망(GAN).2

**AI, ML, DL의 주요 차이점**

| 범주 | 정의 | 관계 | 데이터 특성 추출 | 학습 방식 | 데이터/연산 요구량 | 주요 기술/알고리즘 예시 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **인공지능 (AI)** | 인간의 지능이 필요한 방식으로 추론, 학습, 행동하는 컴퓨터/기계 구축 과학 분야. 광범위한 기술 집합. | 최상위 개념 | \- | \- | \- | 자연어 처리, 추천 시스템, 로봇 자동화 |
| **머신러닝 (ML)** | 명시적 프로그래밍 없이 데이터로부터 자율적으로 학습하고 개선하는 AI 하위 집합. 패턴 인식 및 예측. | AI의 하위 집합 | 수동 개입 필요 (엔지니어 식별) | 인간 개입 필요 | 적음 / CPU 기반 서버 | 선형 회귀, SVM, 결정 트리, K-평균 클러스터링 |
| **딥러닝 (DL)** | 인공 신경망(3개 이상 레이어)을 사용하여 정보를 처리/분석하는 ML 하위 집합. 인간 뇌 활동에서 영감. | ML의 하위 집합 | 자동 추출 (사람 개입 덜 필요) | 자체 오류를 통해 학습 | 많음 / GPU와 같은 강력한 칩 필요 | CNN, RNN, LSTM, GAN |

**배경 지식: AI의 역사적 발전 과정**

인공지능의 역사는 초기 아이디어부터 현재의 전성기까지 여러 부침을 겪으며 발전해왔습니다. 과거의 인공지능은 사람이 일일이 규칙을 정해주는 '규칙 기반 시스템(rule-based system)'으로 작동했습니다.6 예를 들어, "이렇게 생긴 건 고양이야"와 같은 규칙을 직접 만들어서 시스템을 작동시켰습니다.6 그러나 인터넷의 확산으로 데이터의 양이 기하급수적으로 늘어나고 복잡성이 증가하면서, 모든 규칙을 사람이 수동으로 정의하는 방식은 한계에 부딪혔습니다.6 이러한 규칙 기반 시스템의 한계는 컴퓨터가 '경험으로부터 학습'하는 새로운 패러다임, 즉 머신러닝의 필요성을 촉발했습니다. 이 전환은 인공지능 발전의 근본적인 변화를 나타내며, 현재 인공지능의 성공이 대규모 데이터와 학습 알고리즘의 결합에서 비롯되었음을 이해하는 중요한 배경이 됩니다.

머신러닝의 탄생은 1950년대로 거슬러 올라갑니다. 이 시기에 컴퓨터 과학자들은 기계가 인간처럼 학습하고 사고할 수 있는 방법을 모색하기 시작했습니다.7 1959년, IBM의 아서 사무엘(Arthur Samuel)은 '머신러닝'이라는 용어를 처음 제안하며 체커 게임 프로그램을 통해 컴퓨터가 경험으로부터 학습하는 초기 사례를 보여주었습니다.7 그의 프로그램은 과거 게임의 승패 결과를 기록하고 이를 후속 학습에 반영하여 성능을 개선했으며, 이는 오늘날 머신러닝의 핵심 아이디어와 맞닿아 있습니다.9

딥러닝의 시초는 1943년 워렌 맥컬럭(Warren McCulloch)과 월터 피츠(Walter Pitts)가 인간의 뇌 신경세포 구조를 수학적으로 모델링한 인공 신경망(Artificial Neural Networks) 개념을 제안하면서 시작되었습니다.8 1958년 프랭크 로젠블라트(Frank Rosenblatt)는 이 아이디어를 바탕으로 최초의 인공 신경망 모델인 퍼셉트론(Perceptron)을 개발했습니다. 퍼셉트론은 간단한 뉴런 모델을 컴퓨터에 적용하여 입력 데이터에 기반한 결정을 내리는 기초적인 신경망이었습니다.7 그러나 1969년 마빈 민스키(Marvin Minsky)와 시모어 페퍼트(Seymour Papert)가 『Perceptrons』라는 저서에서 단층 퍼셉트론이 XOR 연산과 같은 비선형 문제를 해결할 수 없음을 밝히면서 8, 인공 신경망 연구는 첫 번째 'AI 겨울(AI Winter)'이라는 암흑기를 맞았습니다.8

1980년대에 들어서 컴퓨터 기술의 발전과 함께 머신러닝 연구가 다시 활발해졌습니다.7 다층 퍼셉트론(MLP)과 같은 복잡한 네트워크 모델이 개발되었고, 1974년 폴 워보스(Paul Werbos)가 처음 제안하고 1986년 럼멜하트(Rumelhart), 힌튼(Hinton), 윌리엄스(Williams)에 의해 널리 개발된 역전파 알고리즘(Backpropagation Algorithm) 덕분에 더 깊은 네트워크의 학습이 가능해졌습니다.7 이러한 발전은 신경망이 더 복잡한 패턴을 인식하고 학습할 수 있게 만들었습니다.7

그러나 1990년대에 접어들면서 실생활 문제 데이터의 차원이 증가하고 구조가 복잡해지면서 딥러닝 개념들은 당시 기술력 부족으로 실질적인 한계를 맞이하며 다시 암흑기를 겪었습니다.8 이는 주로 '기울기 소실(Vanishing gradient)', '과적합(Overfitting)', 그리고 당시 하드웨어로는 감당하기 어려운 '느린 학습 시간' 등의 문제 때문이었습니다.8 이 시기에는 서포트 벡터 머신(SVM), 랜덤 포레스트, 부스팅과 같은 다양한 머신러닝 알고리즘들이 인공지능의 대세를 이루었습니다.

딥러닝의 '깊이'는 단순히 신경망 층의 수가 많다는 것을 넘어, 과거 인공 신경망의 암흑기를 초래했던 근본적인 문제들을 해결하기 위한 지속적인 알고리즘 연구의 결과입니다. 2006년 제프리 힌튼(Geoffrey Hinton) 연구진이 비지도 학습 방식의 선학습법(Pre-training)을 적용한 RBM(Restricted Boltzmann Machine) 모델을 발표하며 신경망의 고질적인 문제들을 일부 해결했고, 딥러닝이 다시 주목받기 시작했습니다.8 이후 2010년 힌튼 교수가 ReLU(Rectified Linear Unit) 활성화 함수를 제안하여 기울기 소실 문제를 완화하고 선학습의 필요성을 줄였으며 8, 2012년에는 DropOut이라는 정규화 방법을 제안하여 과적합 문제를 해결하는 데 기여했습니다.8

결정적인 전환점은 2012년 알렉스 크리제브스키(Alex Krizhevsky), 일리야 수츠케버(Ilya Sutskever), 그리고 제프리 힌튼이 개발한 AlexNet(CNN)이 이미지 인식 대회(ILSVRC)에서 압도적인 성능으로 우승하며 '알렉스넷 쇼크'를 일으킨 사건입니다.8 이 성공은 단순히 GPU 사용과 빅데이터의 힘뿐만 아니라, ReLU 활성화 함수, Dropout과 같은 정규화 기법 등 수십 년간 축적된 알고리즘적 발전이 결합되어 가능했습니다. 이는 딥러닝의 현재 성공이 단순히 컴퓨팅 자원의 증가 덕분만이 아니라, 복잡한 문제를 해결하기 위한 심오한 이론적, 알고리즘적 돌파구의 결과임을 시사합니다.

AlexNet 이후, 딥러닝은 전성기를 맞이하며 폭발적으로 발전했습니다.8 1997년 제프 호크라이터(Sepp Hochreiter)와 위르겐 슈미트후버(Jürgen Schmidhuber)가 제안한 LSTM(Long Short-Term Memory)은 순환 신경망(RNN)의 장기 의존성 문제를 해결하며 시퀀스 데이터 처리에 혁신을 가져왔고 8, 2014년 이안 굿펠로우(Ian Goodfellow)가 제안한 GAN(Generative Adversarial Network)은 실제와 같은 가짜 데이터를 생성하는 기술로 큰 주목을 받았습니다.8 이러한 기술적 발전과 함께, 인공지능은 점차 다양한 분야에서 응용되기 시작했고, 오늘날에는 거의 모든 산업에서 중요한 역할을 담당하고 있습니다.7

### **2.2. 사례: AI/ML/DL 기술의 실제 적용 사례**

AI, 머신러닝, 딥러닝 기술은 단순한 학술적인 개념을 넘어, 우리의 일상생활과 산업 전반에 걸쳐 혁신적인 변화를 가져오고 있습니다. 이 과정의 목표가 "상용 AI 소프트웨어 개발 능력" 1임을 고려할 때, 이러한 사례들은 학습자가 배울 기술이 실제 시장에서 어떤 가치를 창출할 수 있는지에 대한 강력한 동기 부여가 됩니다. 이는 이론적 지식이 실제 문제 해결 능력으로 이어지는 다리 역할을 합니다.

**AI의 광범위한 적용**

인공지능은 데이터 분석, 예상 및 예측, 자연어 처리, 추천 시스템, 머신 자동화, 지능형 데이터 검색 등 다양한 비즈니스 및 일상 생활 분야에 적용됩니다.2

**머신러닝 사례**

* **지도 학습:**  
  * **스팸 메일 필터링:** 과거의 스팸/정상 메일 데이터를 학습하여 새로운 메일을 자동으로 분류합니다.  
  * **이미지 분류:** 라벨링된 사과 그림 데이터를 학습하여 새로운 이미지에서 사과를 인식합니다.2  
  * **신용카드 사기 탐지:** 정상 거래와 사기 거래 데이터를 학습하여 의심스러운 거래를 식별합니다.  
  * Python의 사이킷런(Scikit-Learn)과 같은 라이브러리가 대표적으로 활용됩니다.4  
* **비지도 학습:**  
  * **고객 세분화:** 구매 이력이나 행동 패턴을 기반으로 고객을 그룹으로 분류하여 맞춤형 마케팅 전략 수립에 활용됩니다.2  
  * **이상 탐지:** 정상 범주에서 벗어나는 데이터 패턴을 자동으로 식별하여 시스템 오류나 네트워크 침입 등을 감지합니다.2  
* **강화 학습:**  
  * **로봇 제어:** 로봇이 시행착오를 통해 최적의 움직임을 학습하여 복잡한 작업을 수행할 수 있도록 합니다.  
  * **게임 플레이:** Google의 AlphaGo는 바둑 규칙에 대한 사전 지식 없이도 강화 학습을 통해 스스로 학습하여 인간 챔피언을 이겼습니다.2 이는 인공지능의 지능적 능력을 보여주는 대표적인 사례입니다.

**딥러닝 사례**

* **이미지 및 음성 인식:** 스마트폰의 얼굴 인식 잠금 해제, 음성 비서(Siri, Google Assistant)의 음성 명령 처리, 의료 영상(X-ray, MRI) 분석을 통한 질병 진단 보조 등에 활용됩니다.2  
* **객체 감지:** 자율 주행 자동차가 도로 위의 보행자 및 차량을 실시간으로 감지하거나, CCTV 영상에서 특정 객체(예: 침입자)를 식별하는 데 사용됩니다.2  
* **자연어 처리 (NLP):** 기계 번역(Google 번역), 챗봇 및 대화형 인공지능(고객 문의 응대), 온라인 리뷰에서 긍정/부정 감성을 파악하는 감성 분석 등에 활용됩니다.2  
* **우편번호 자동 인식:** 얀 르쿤(Yann LeCun)이 개발한 미국 우체국을 위한 우편번호 자동 인식 프로그램은 MNIST 데이터셋을 활용한 CNN의 초기 성공 사례로, 현재까지도 우편물 분류 시스템에 활용되고 있습니다.13 이는 딥러닝이 실용적인 자동화 능력을 제공하는 대표적인 예시입니다.

### **2.3. 학습내용 확인: 개념 이해도 점검 문제**

1. **개념 정의:** 인공지능(AI), 머신러닝(ML), 딥러닝(DL)의 개념을 각각 정의하고, 이들 간의 포함 관계를 설명하시오.  
2. **차이점 분석:** 머신러닝과 딥러닝의 주요 차이점 (예: 데이터 특성 추출 방식, 학습 방식, 필요한 데이터 및 연산 능력)을 비교 설명하시오.  
3. **역사적 사건:** AI, 머신러닝, 딥러닝 역사에서 중요한 전환점(예: 퍼셉트론의 한계, 역전파 알고리즘의 등장, AlexNet 쇼크 등)을 최소 2가지 이상 언급하고, 각 사건이 해당 분야 발전에 미친 영향을 설명하시오.  
4. **사례 연결:** 다음 실제 사례들이 AI, 머신러닝, 딥러닝 중 어떤 기술에 해당하는지 분류하고, 그 이유를 간략히 설명하시오.  
   * a) 스팸 메일 필터링 (과거 스팸 메일 데이터로 학습)  
   * b) 자율 주행 자동차의 실시간 객체 인식  
   * c) 바둑 게임에서 인간 챔피언을 이긴 AlphaGo

## **3\. 구글 Colab 사용법**

클라우드 기반의 개발 환경인 Google Colab의 주요 기능과 사용법을 익혀, 향후 딥러닝 모델 개발 및 실습에 필요한 환경을 구축합니다.

### **3.1. 이론강의: Google Colab의 이해와 활용**

Colaboratory(줄여서 'Colab')는 브라우저 내에서 Python 스크립트를 작성하고 실행할 수 있는 플랫폼입니다.14 이는 Jupyter 메모장(노트북)을 기반으로 하며, 실행 코드와 서식 있는 텍스트, 이미지, HTML, LaTeX 등을 하나의 문서로 통합하여 활용할 수 있습니다.14

Google Colab의 주요 장점은 다음과 같습니다:

* **구성 불필요 (Zero Configuration):** Colab은 별도의 설치나 복잡한 설정 없이 바로 사용할 수 있어, 개발 환경 구축에 드는 시간을 절약하고 즉시 코딩을 시작할 수 있게 합니다.14  
* **무료 GPU/TPU 제공:** 딥러닝 모델 학습에는 고성능 컴퓨팅 자원이 필수적입니다. Colab은 고가의 하드웨어 없이도 강력한 GPU 또는 TPU를 무료로 사용할 수 있게 합니다.14 이는 Google 클라우드 서버에서 코드를 실행하므로 사용자 컴퓨터 성능과 관계없이 Google 하드웨어 성능을 활용할 수 있게 합니다.14 과거 딥러닝 연구의 '암흑기' 8의 한계 중 하나였던 '느린 학습시간'과 '하드웨어 부담' 8은 고성능 컴퓨팅 자원의 부족에서 비롯되었습니다. Colab은 이러한 문제를 해결하여 개인 개발자나 소규모 팀도 고가의 장비 없이 딥러닝 모델을 개발하고 실험할 수 있게 함으로써, AI 개발의 '민주화'에 크게 기여하고 있습니다. 이는 본 과정의 목표인 "상용 AI 소프트웨어 개발 능력" 1을 달성하는 데 필수적인 기반 환경을 제공합니다.  
* **간편한 공유 및 협업:** Colab 메모장을 간편하게 공유하여 동료나 친구들이 댓글을 달거나 수정하도록 할 수 있어, 팀 프로젝트나 교육 환경에서 협업 효율성을 높입니다.14  
* **Google Drive 통합:** 데이터 세트와 모델을 Google Drive에 쉽게 저장하고 액세스할 수 있으며, Colab 메모장 자체도 Google Drive 계정에 저장됩니다.14 이는 작업의 지속성을 보장하고 대용량 데이터를 효율적으로 관리할 수 있게 합니다.  
* **머신러닝 커뮤니티에서 널리 사용:** Colab은 TensorFlow 시작, 신경망 개발 및 학습, TPU 실험, AI 연구 보급, 튜토리얼 생성 등 다양한 머신러닝 분야에서 널리 쓰이고 있어, 관련 자료를 찾고 학습하는 데 용이합니다.14

### **3.2. 실습 제안: Colab 개발 환경 구축 및 기본 조작**

**GPU 런타임 설정 방법**

딥러닝 모델 학습 시 GPU를 활용하면 연산 속도를 크게 향상시킬 수 있습니다. Colab에서 GPU 런타임을 설정하는 단계는 다음과 같습니다:

1. Colab 노트북을 엽니다.  
2. 상단의 '런타임' 탭을 클릭합니다.15  
3. 드롭다운 메뉴에서 '런타임 유형 변경'을 선택합니다.15  
4. '하드웨어 가속기' 드롭다운 메뉴에서 'GPU'를 선택하고 '저장' 버튼을 클릭합니다.15  
5. 우측 상단에 '연결됨' 표시와 함께 GPU 환경에 성공적으로 연결되었는지 확인할 수 있습니다.15  
6. \!nvidia-smi 명령어를 셀에 입력하고 실행하여 현재 할당된 GPU의 사양을 확인할 수 있습니다.19

**활용 팁:**

* 메모리 사용량 최적화를 위해 트레이닝 중 이미지 크기 또는 배치 크기를 줄이거나, tf.config.experimental.set\_memory\_growth(gpu, True)와 같은 코드를 사용하여 특정 GPU에 대한 메모리 증가를 활성화할 수 있습니다.17  
* Colab은 세션 시간 제한이 있으므로, 진행 상황을 놓치지 않도록 모델과 결과를 자주 저장하는 것이 중요합니다.17

**Google Drive 연동 및 파일 시스템 접근**

Colab에서 Google Drive에 저장된 데이터나 모델에 접근하려면 Drive를 마운트해야 합니다.

1. Colab 노트북의 코드 셀에 다음 파이썬 코드를 입력하고 실행합니다 16:  
   ```python
   from google.colab import drive  
   drive.mount('/content/gdrive')
   ```
2. 실행 후 나타나는 링크를 클릭하여 Google 계정을 선택하고 인증 절차를 완료합니다.
3. 발급된 인증 키(authorization code)를 Colab의 입력창에 붙여넣고 엔터를 누르면 Google Drive가 Colab 환경에 마운트됩니다.
4. 마운트 성공 후, /content/gdrive/MyDrive 경로를 통해 자신의 Google Drive 폴더에 접근할 수 있습니다.23 외부 파일을 사용하고 싶을 경우, Google Drive에 업로드하여 이 경로를 통해 접근하면 됩니다.24

**주요 Colab 명령어 및 단축키 활용**

Colab의 효율적인 사용을 위해 자주 사용되는 명령어와 단축키를 숙지하는 것이 중요합니다. 이는 개발 워크플로우를 최적화하고 생산성을 높이는 데 기여합니다.

| 분류 | 명령어/단축키 | 설명 |
| :---- | :---- | :---- |
| **셀 실행** | Ctrl \+ Enter | 현재 셀을 실행하고 커서를 해당 셀에 유지합니다 (결과 값만 확인 시 유용).25 |
|  | Shift \+ Enter | 현재 셀을 실행하고 커서를 다음 셀로 이동합니다 (여러 셀을 빠르게 실행할 때 유용).25 |
|  | Alt \+ Enter | 현재 셀을 실행하고 아래에 새로운 코드 셀을 삽입한 후 커서를 삽입된 셀로 이동합니다 (다음 작업 공간이 필요할 때 유용).25 |
| **셀 삽입/삭제** | Ctrl \+ M A | 현재 셀 위에 새로운 코드 셀을 삽입합니다.25 |
|  | Ctrl \+ M B | 현재 셀 아래에 새로운 코드 셀을 삽입합니다.25 |
|  | Ctrl \+ M D | 현재 셀을 삭제합니다.25 |
| **셀 유형 변경** | Ctrl \+ M Y | 현재 셀을 코드 셀로 변경합니다.25 |
|  | Ctrl \+ M M | 현재 셀을 마크다운 셀로 변경합니다.25 |
| **파일 시스템/패키지 관리** | \!pip install \[패키지명\] | 필요한 파이썬 패키지를 설치합니다.24 |
|  | \!ls | 현재 작업 디렉토리의 파일 목록을 확인합니다.21 |
|  | \!pwd | 현재 작업 디렉토리의 경로를 확인합니다.21 |
|  | from google.colab import drive; drive.mount('/content/gdrive') | Google Drive를 Colab 환경에 마운트합니다.16 |

### **3.3. 학습내용 확인: Colab 활용 능력 점검 문제**

1. **GPU 설정:** Google Colab에서 GPU 런타임을 설정하는 단계를 순서대로 설명하고, 설정이 올바르게 되었는지 확인하는 명령어를 제시하시오.  
2. **드라이브 연동:** Colab 노트북에서 Google Drive를 연동하여 Drive 내의 파일에 접근하는 파이썬 코드를 작성하고, 연동된 Drive의 기본 경로를 설명하시오.  
3. **기본 조작:** 다음 작업을 수행하기 위한 Colab 단축키 또는 명령어를 작성하시오.  
   * a) 현재 셀을 실행하고 다음 셀로 커서 이동  
   * b) 현재 셀 아래에 새로운 코드 셀 삽입  
   * c) requests 라이브러리 설치  
   * d) 현재 작업 디렉토리의 파일 목록 확인

## **4\. Numpy, Pandas, Matplotlib 사용법 실습**

파이썬 기반 데이터 과학 및 AI 개발의 핵심 라이브러리인 Numpy, Pandas, Matplotlib의 기본적인 사용법을 익히고, 데이터 처리 및 시각화의 기초를 다집니다. 이 세 가지 라이브러리는 단순히 개별적인 도구가 아니라, 파이썬 기반 데이터 과학 및 AI 개발의 필수적인 '기본 도구 모음'을 형성합니다. 이들은 데이터 수집부터 전처리, 분석, 시각화에 이르는 전체 데이터 파이프라인을 가능하게 하며, 이는 어떤 AI/ML 모델을 개발하든 필수적으로 거쳐야 하는 과정입니다. 이 라이브러리들에 대한 숙련은 효율적인 데이터 핸들링과 통찰력 도출 능력으로 직결되어, 향후 과정에서 다룰 복잡한 딥러닝 모델 학습 및 서비스 개발의 성공 여부를 좌우합니다.

### **4.1. 이론강의: 데이터 처리 및 시각화의 기초**

**Numpy (Numerical Python)**

Numpy는 과학 연산을 위한 파이썬의 핵심 라이브러리입니다.26 특히 다차원 배열 객체인

ndarray를 효율적으로 다루는 기능을 제공하며 26, 이는 파이썬의 기본

list 객체보다 훨씬 더 많은 데이터를 빠르게 처리할 수 있도록 개선되었습니다.26

ndarray 객체의 모든 요소는 int32, float64 등과 같이 동일한 데이터 형을 가지므로 배열 연산 시 처리 속도가 빠르다는 장점이 있습니다.27

Numpy의 핵심적인 강점은 '벡터화(Vectorization)'된 연산을 지원한다는 점입니다.28 이는 내부적으로 C/포트란과 같은 저수준 언어로 최적화된 연산을 수행하여, 대규모 데이터셋에 대한 복잡한 계산을 파이썬 루프보다 훨씬 빠르게 처리할 수 있게 합니다. Numpy는 강력한 N차원 배열 객체, 정교한 브로드캐스팅(Broadcast) 기능, C/C++ 및 포트란 코드 통합 도구, 유용한 선형 대수학, 푸리에 변환 및 난수 기능 등을 제공합니다.26

**Pandas (Python Data Analysis Library)**

Pandas는 데이터 처리 및 분석을 위한 라이브러리로, 특히 정형 데이터(표 형태의 데이터)를 다루는 데 강력한 기능을 제공합니다.

* **주요 데이터 구조:**  
  * **Series:** 1차원 데이터 배열을 나타내는 객체로, 인덱스(index)와 값(value)으로 구성됩니다. 각 값은 인덱스에 해당하는 레이블로 접근할 수 있습니다.29  
  * **DataFrame:** 서로 같거나 다른 데이터형의 여러 개의 열에 대하여 복수 개의 성분으로 구성된 '표와 같은 형태'의 2차원 자료 구조입니다.30 데이터 분석의 가장 기본적인 단위로 활용되며, 관계형 데이터베이스의 테이블이나 스프레드시트와 유사한 구조를 가집니다.

Pandas의 핵심 기능에는 CSV, Excel 등 다양한 형태의 파일에서 데이터를 효율적으로 불러오고 저장하는 기능이 포함되며 29, 데이터 탐색, 선택, 필터링, 정렬, 그룹화 등 복잡한 데이터 조작을 쉽게 수행할 수 있도록 돕습니다. 데이터셋에 값이 존재하지 않을 때는

NaN(Not a Number)으로 표시되며, 이는 연산에 영향을 미치므로 적절한 처리 방법을 인지하는 것이 중요합니다.30

**Matplotlib**

Matplotlib은 Python에서 데이터를 시각화해주는 가장 기본적인 패키지입니다.32 빅데이터들을 분석 시 한눈에 파악하기 어려울 때, 시각화를 통해 데이터를 직관적으로 이해하고 분석에 활용할 수 있도록 돕는 것이 주요 목적입니다.32

Matplotlib은 pyplot이라는 서브패키지를 주로 사용하며, 선 그래프, 산점도, 히스토그램, 막대 그래프, 파이 차트 등 다양한 2D 그래프를 그릴 수 있습니다.29 그래프의 스타일(색깔, 점 모양, 선 스타일), 축 범위 지정, 여러 그래프를 한 번에 그리는 서브플롯(subplot) 기능 등도 제공하여 32 데이터의 다양한 측면을 효과적으로 탐색하고 표현할 수 있게 합니다. 시각화는 단순히 그래프를 그리는 것을 넘어, 복잡한 데이터 속에서 '패턴과 이상치'를 직관적으로 발견하고, 모델의 학습 과정과 결과를 '진단'하며 '이해'하는 데 필수적인 도구입니다. 이러한 효율성과 통찰력은 AI 서비스 개발의 성공에 직접적인 영향을 미칩니다.

### **4.2. 실습 제안: 핵심 라이브러리 활용**

**Numpy 실습**

Numpy는 다차원 배열 ndarray를 기반으로 효율적인 수치 연산을 제공합니다.

**Numpy 핵심 함수 요약**

| 분류 | 함수/메서드 | 설명 | 예시 (코드 및 결과) |
| :---- | :---- | :---- | :---- |
| **배열 생성** | np.array() | 파이썬 리스트로부터 ndarray 생성 | data \= ; arr \= np.array(data) array() |
|  | np.zeros() | 모든 요소가 0인 배열 생성 | np.zeros((2, 3)) \[\[0., 0., 0.\], \[0., 0., 0.\]\] |
|  | np.ones() | 모든 요소가 1인 배열 생성 | np.ones((2, 3), dtype=int) \[, \] |
|  | np.arange() | 특정 범위 내에서 균일한 간격으로 데이터 생성 | np.arange(10) \`\` |
| **배열 속성** | .ndim | 배열의 차원 수 | arr.ndim 1 |
|  | .shape | 배열의 각 차원 크기 (튜플) | arr.shape (3,) |
|  | .dtype | 배열 요소의 데이터 타입 | arr.dtype dtype('int32') |
|  | .astype() | 배열의 데이터 타입 변경 | arr\_float \= arr.astype(np.float64) dtype('float64') |
| **기본 연산** | \+, \-, \*, / | 요소별 사칙연산 | x \= np.array(); y \= np.array(); x \+ y \`\` |
|  | .dot() 또는 @ | 행렬 곱셈 (내적) | x \= np.array(\[,\]); y \= np.array(\[,\]); x.dot(y) \[, \] |
| **통계 연산** | np.sum() | 배열의 모든 요소 합 (axis 지정 가능) | np.sum(x) 10 |
|  | np.mean() | 배열 요소의 평균 | np.mean(x) 2.5 |
|  | np.max(), np.min() | 배열 요소의 최대/최소값 | np.max(x) 4 |
|  | np.std() | 배열 요소의 표준편차 | np.std(x) 1.118... |
|  | .T | 전치 행렬 (Transpose) | x.T \[, \] |
| **인덱싱/슬라이싱** | arr\[idx\] | 특정 인덱스 요소 접근 | arr \= np.arange(10); arr 5 |
|  | arr\[start:end\] | 슬라이싱 (뷰 반환) | arr\[5:8\] \`\` |
|  | arr.copy() | 배열 복사 (뷰가 아닌 독립 객체) | arr\_copy \= arr\[5:8\].copy() \`\` |
| **브로드캐스팅** | 스칼라/벡터 연산 | Shape이 다른 배열 간 연산 (자동 확장) | x \= np.array(\[,\]); x \+ 2 \[, \] |

**Pandas 실습**

Pandas는 데이터프레임(DataFrame)을 중심으로 데이터를 효율적으로 조작하고 분석합니다.

**Pandas DataFrame/Series 핵심 기능 요약**

| 분류 | 함수/메서드 | 설명 | 예시 (코드 및 결과) |
| :---- | :---- | :---- | :---- |
| **객체 생성** | pd.DataFrame() | 딕셔너리, 리스트 등으로 DataFrame 생성 | data \= {'col1': , 'col2': }; df \= pd.DataFrame(data) col1 col2 0 1 3 1 2 4 |
|  | pd.Series() | 1차원 Series 객체 생성 | s \= pd.Series() 0 1 1 2 2 3 |
| **데이터 불러오기** | pd.read\_csv() | CSV 파일 불러오기 | df \= pd.read\_csv('data.csv') |
|  | pd.read\_excel() | Excel 파일 불러오기 | df \= pd.read\_excel('data.xlsx') |
| **데이터 탐색** | df.head() | DataFrame 상위 5행 (또는 지정 개수) 출력 | df.head(3) |
|  | df.tail() | DataFrame 하위 5행 (또는 지정 개수) 출력 | df.tail(2) |
|  | df.info() | DataFrame의 요약 정보 (컬럼별 타입, Non-null 개수) | df.info() |
|  | df.describe() | 숫자형 컬럼의 기술 통계량 (평균, 표준편차 등) | df.describe() |
|  | df\['col'\].value\_counts() | 특정 컬럼의 고유 값별 빈도수 계산 | df\['성별'\].value\_counts() |
| **데이터 선택/필터링** | df\['col'\] | 단일 컬럼 선택 (Series 반환) | df\['나이'\] |
|  | df\[\['col1', 'col2'\]\] | 여러 컬럼 선택 (DataFrame 반환) | df\[\['이름', '나이'\]\] |
|  | df\[df\['col'\] \> value\] | 조건에 따른 행 필터링 | df\[df\['나이'\] \> 30\] |
| **데이터 조작** | df\['new\_col'\] \= value | 새 컬럼 생성 및 값 대입 | df\['출생지'\] \= '한국' |
|  | df\['col'\] \= new\_value | 기존 컬럼 값 수정 | df\['출생지'\] \= '서울' |
|  | df.sort\_values(by='col') | 특정 컬럼 기준으로 정렬 | df.sort\_values(by='나이', ascending=False) |
|  | df.drop('col', axis=1) | 컬럼 삭제 (axis=0은 행) | df.drop('이름', axis=1) |
|  | df.groupby('col').mean() | 특정 컬럼으로 그룹화 및 집계 | df.groupby('성별')\['나이'\].mean() |
|  | df.apply(func) | DataFrame 또는 Series에 함수 적용 | df\['나이'\].apply(lambda x: '성인' if x \>= 19 else '미성년') |

**Matplotlib 실습**

Matplotlib은 데이터를 시각적으로 표현하여 패턴과 추세를 쉽게 파악할 수 있도록 돕습니다.

```Python

import matplotlib.pyplot as plt  
import numpy as np  
import pandas as pd

\# 1\. 선 그래프 (Line Plot)  
\# 간단한 선 그래프 그리기  
x \= np.linspace(0, 10, 100) \# 0부터 10까지 100개의 균일한 간격의 숫자 생성  
y \= np.sin(x)  
plt.plot(x, y)  
plt.title('Sine Wave') \# 그래프 제목  
plt.xlabel('X-axis') \# X축 레이블  
plt.ylabel('Y-axis') \# Y축 레이블  
plt.grid(True) \# 그리드 표시  
plt.show()

\# 여러 개의 선 그래프를 한 차트에 그리기  
y2 \= np.cos(x)  
plt.plot(x, y, label='sin(x)', color='blue', linestyle='-') \# 색상, 선 스타일 지정  
plt.plot(x, y2, label='cos(x)', color='red', linestyle='--')  
plt.title('Sine and Cosine Waves')  
plt.xlabel('X-axis')  
plt.ylabel('Y-axis')  
plt.legend() \# 범례 표시  
plt.grid(True)  
plt.show()

\# 2\. 산점도 (Scatter Plot)  
\# 무작위 데이터를 이용한 산점도  
np.random.seed(0) \# 재현성을 위한 시드 설정  
x\_scatter \= np.random.rand(50) \* 10  
y\_scatter \= np.random.rand(50) \* 10  
sizes \= np.random.rand(50) \* 800 \+ 100 \# 점 크기  
colors \= np.random.rand(50) \# 점 색상 (컬러맵 사용)

plt.scatter(x\_scatter, y\_scatter, s=sizes, c=colors, alpha=0.7, cmap='viridis') \# s:크기, c:색상, alpha:투명도, cmap:컬러맵  
plt.title('Scatter Plot Example')  
plt.xlabel('Feature 1')  
plt.ylabel('Feature 2')  
plt.colorbar(label='Color Value') \# 컬러바 표시  
plt.show()

\# 3\. 히스토그램 (Histogram)  
\# 정규 분포 데이터를 이용한 히스토그램  
data\_hist \= np.random.randn(1000) \# 표준 정규 분포에서 1000개 데이터 생성  
plt.hist(data\_hist, bins=30, edgecolor='black', alpha=0.7) \# bins: 막대 개수  
plt.title('Histogram of Random Data')  
plt.xlabel('Value')  
plt.ylabel('Frequency')  
plt.show()

\# 4\. 막대 그래프 (Bar Plot)  
\# 카테고리별 데이터 시각화  
categories \=  
values \= 

plt.bar(categories, values, color=\['skyblue', 'lightcoral', 'lightgreen', 'gold'\])  
plt.title('Bar Plot of Categories')  
plt.xlabel('Category')  
plt.ylabel('Value')  
plt.show()

\# 5\. 파이 차트 (Pie Chart)  
\# 비율 데이터 시각화  
labels \=  
sizes \=   
explode \= (0, 0.1, 0, 0) \# Banana 조각만 분리

plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)  
plt.title('Fruit Distribution')  
plt.axis('equal') \# 원형 유지  
plt.show()
```
### **4.3. 학습내용 확인: 라이브러리 활용 능력 점검 문제**

1. **Numpy 배열 생성 및 연산:**  
   * a) 3행 4열의 모든 요소가 0으로 채워진 Numpy 배열을 생성하는 코드를 작성하시오.  
   * b) 두 개의 2x2 Numpy 배열 A \= np.array(\[, \])와 B \= np.array(\[, \])를 정의하고, 이 두 배열의 요소별 곱셈(element-wise multiplication)과 행렬 곱셈(matrix multiplication)을 수행하는 코드를 각각 작성하시오.  
2. **Pandas DataFrame 조작:**  
   * a) 다음 딕셔너리를 사용하여 DataFrame을 생성하는 코드를 작성하고, 생성된 DataFrame의 상위 3행을 출력하는 코드를 작성하시오.  
     Python  
     data \= {'Name':,  
             'Age': ,  
             'City':}

   * b) 위에서 생성한 DataFrame에서 'Age' 컬럼의 평균값을 계산하는 코드를 작성하시오.  
   * c) 'Age'가 30 이상인 모든 행을 필터링하여 출력하는 코드를 작성하시오.  
3. **Matplotlib 시각화:**  
   * a) x 값을 0부터 2π까지 100개의 균일한 간격으로 생성하고, y 값을 sin(x)로 하여 선 그래프를 그리는 코드를 작성하시오. 그래프에 'Sine Wave'라는 제목과 'X-axis', 'Y-axis' 레이블을 추가하시오.  
   * b) 임의의 50개 데이터를 생성하여 산점도(Scatter Plot)를 그리는 코드를 작성하시오. 점의 크기와 색상을 임의로 다르게 표현하는 옵션을 추가하시오.

## **5\. 종합 학습 내용 확인 및 다음 단계 안내**

### **5.1. 1주 1일차 핵심 요약**

1주 1일차 강의는 AI 서비스 개발의 여정을 시작하는 중요한 첫걸음이었습니다. 학습자는 인공지능, 머신러닝, 딥러닝의 개념과 이들 간의 계층적 관계를 명확히 이해하고, 각 기술이 어떻게 발전해왔는지 역사적 흐름을 통해 파악했습니다. 특히, 규칙 기반 시스템의 한계에서 데이터 기반 학습으로의 패러다임 전환과 딥러닝의 암흑기를 극복하게 한 알고리즘적 돌파구(예: 역전파, ReLU, Dropout, AlexNet)의 중요성을 인지했습니다.

또한, 클라우드 기반의 개발 환경인 Google Colab의 사용법을 익히며, 무료 GPU/TPU 활용, Google Drive 연동, 그리고 효율적인 코드 작성을 위한 단축키 및 명령어 활용법을 실습했습니다. Colab이 AI 개발의 진입 장벽을 낮추고 민주화에 기여하는 핵심적인 도구임을 이해했습니다.

마지막으로, 파이썬 데이터 과학의 핵심 라이브러리인 Numpy, Pandas, Matplotlib의 기본 사용법을 익혔습니다. Numpy를 통해 효율적인 다차원 배열 연산의 중요성을, Pandas를 통해 정형 데이터의 효과적인 처리 및 조작 방법을, 그리고 Matplotlib을 통해 데이터 시각화의 중요성과 다양한 그래프 작성법을 학습했습니다. 이 세 라이브러리는 데이터 수집부터 전처리, 분석, 시각화에 이르는 AI 개발의 필수적인 데이터 파이프라인을 구축하는 데 핵심적인 역할을 합니다.

### **5.2. 질의응답 및 다음 강의 예고**

오늘 학습한 내용에 대해 궁금한 점이 있다면 자유롭게 질문해 주시기 바랍니다.

다음 1주 2일차 강의에서는 '파이토치 기본'에 대해 학습할 예정입니다.1 텐서(Tensor) 자료구조의 이해와 연산, 자동 미분(Autograd)과 경사하강법(Gradient Descent)의 원리, 그리고 파이토치 기반 선형 회귀(Linear Regression) 모델 구현을 다루며, 오늘 배운 개발 환경과 파이썬 라이브러리 지식을 바탕으로 본격적인 딥러닝 모델링의 기초를 다지게 될 것입니다.1 오늘 학습한 내용들을 복습하여 다음 강의에 대비하시기를 권장합니다.

#### **참고 자료**

1. 교육과정-v2  
2. 딥 러닝과 머신러닝 비교 | Google Cloud, 6월 30, 2025에 액세스, [https://cloud.google.com/discover/deep-learning-vs-machine-learning?hl=ko](https://cloud.google.com/discover/deep-learning-vs-machine-learning?hl=ko)  
3. kmong.com, 6월 30, 2025에 액세스, [https://kmong.com/article/1327--%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EB%94%A5%EB%9F%AC%EB%8B%9D-%EC%B0%A8%EC%9D%B4%EC%A0%90-5%EA%B0%80%EC%A7%80\#:\~:text=%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%9D%80%20%EC%BB%B4%ED%93%A8%ED%84%B0%EA%B0%80,%EA%B0%80%EB%A5%B4%EC%B9%98%EB%8A%94%20%EC%9D%B8%EA%B3%B5%20%EC%A7%80%EB%8A%A5%20%EB%B0%A9%EC%8B%9D%EC%9E%85%EB%8B%88%EB%8B%A4.](https://kmong.com/article/1327--%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EB%94%A5%EB%9F%AC%EB%8B%9D-%EC%B0%A8%EC%9D%B4%EC%A0%90-5%EA%B0%80%EC%A7%80#:~:text=%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D%EC%9D%80%20%EC%BB%B4%ED%93%A8%ED%84%B0%EA%B0%80,%EA%B0%80%EB%A5%B4%EC%B9%98%EB%8A%94%20%EC%9D%B8%EA%B3%B5%20%EC%A7%80%EB%8A%A5%20%EB%B0%A9%EC%8B%9D%EC%9E%85%EB%8B%88%EB%8B%A4.)  
4. 인공지능 머신러닝 딥러닝 개념 차이 관계 간단정리\! \- For Data Science \- 티스토리, 6월 30, 2025에 액세스, [https://for-data-science.tistory.com/56](https://for-data-science.tistory.com/56)  
5. 인공지능·머신러닝·딥러닝 차이점은?ㅣ개념부터 차이점까지 총 정리 \- 코드스테이츠, 6월 30, 2025에 액세스, [https://www.codestates.com/blog/content/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EB%94%A5%EB%9F%AC%EB%8B%9D%EA%B0%9C%EB%85%90](https://www.codestates.com/blog/content/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EB%94%A5%EB%9F%AC%EB%8B%9D%EA%B0%9C%EB%85%90)  
6. 03화 2\. 머신러닝과 딥러닝을 알아보자, 6월 30, 2025에 액세스, [https://brunch.co.kr/@@8L27/21](https://brunch.co.kr/@@8L27/21)  
7. \[AI\] 인공지능의 역사 \- 머신러닝이란?(ML) \- yeonjin \- 티스토리, 6월 30, 2025에 액세스, [https://yeonjinj.tistory.com/14](https://yeonjinj.tistory.com/14)  
8. \[Deep Learning\] 딥러닝의 역사 \- Data Science \- 티스토리, 6월 30, 2025에 액세스, [https://lebi.tistory.com/18](https://lebi.tistory.com/18)  
9. 머신러닝을 처음 만든 사람은 누구일까?, 6월 30, 2025에 액세스, [https://brunch.co.kr/@@dTP6/179](https://brunch.co.kr/@@dTP6/179)  
10. 인공지능의 역사를 열어온 인물들, 6월 30, 2025에 액세스, [https://brunch.co.kr/@hvnpoet/66](https://brunch.co.kr/@hvnpoet/66)  
11. 인공지능/역사 \- 나무위키, 6월 30, 2025에 액세스, [https://namu.wiki/w/%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5/%EC%97%AD%EC%82%AC](https://namu.wiki/w/%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5/%EC%97%AD%EC%82%AC)  
12. \[김인중이 전하는 딥러닝의 세계\] \<7\> 딥러닝 역사의 전환점들 \- 한국경제, 6월 30, 2025에 액세스, [https://www.hankyung.com/article/202202179087i](https://www.hankyung.com/article/202202179087i)  
13. 왜 지금인가? \- 딥러닝의 역사, 6월 30, 2025에 액세스, [https://skyil.tistory.com/6](https://skyil.tistory.com/6)  
14. Colab 시작하기 \- Colab \- Google, 6월 30, 2025에 액세스, [https://colab.research.google.com/?hl=ko](https://colab.research.google.com/?hl=ko)  
15. Google Colab에서 GPU 사용 설정하기 \- 큐트리 개발 블로그, 6월 30, 2025에 액세스, [https://devsoyoung.github.io/posts/colab-gpu/](https://devsoyoung.github.io/posts/colab-gpu/)  
16. \[Google Colaboratory\] 코랩으로 GPU, TPU 사용법 \- 개발자 우성우 \- 티스토리, 6월 30, 2025에 액세스, [https://wscode.tistory.com/30](https://wscode.tistory.com/30)  
17. Google Colab으로 YOLO11 프로젝트 가속화하기 \- Ultralytics YOLO 문서 도구, 6월 30, 2025에 액세스, [https://docs.ultralytics.com/ko/integrations/google-colab/](https://docs.ultralytics.com/ko/integrations/google-colab/)  
18. Colab Enterprise 런타임 관리 \- Google Cloud, 6월 30, 2025에 액세스, [https://cloud.google.com/colab/docs/manage-runtimes?hl=ko](https://cloud.google.com/colab/docs/manage-runtimes?hl=ko)  
19. Google Colab 활용 : 사양 확인, 런타임, 파일 저장/다운로드/업로드, 연동 \- Joo.soft, 6월 30, 2025에 액세스, [https://data-jj.tistory.com/16](https://data-jj.tistory.com/16)  
20. GPU 사용하기 \- Colab, 6월 30, 2025에 액세스, [https://colab.research.google.com/github/tensorflow/docs-l10n/blob/master/site/ko/guide/gpu.ipynb?hl=ko](https://colab.research.google.com/github/tensorflow/docs-l10n/blob/master/site/ko/guide/gpu.ipynb?hl=ko)  
21. Colab 초기 설정 튜토리얼 \- Wannabe의 기록 \- 티스토리, 6월 30, 2025에 액세스, [https://wannabe00.tistory.com/entry/Colab-%EC%B4%88%EA%B8%B0-%EC%84%A4%EC%A0%95-%ED%8A%9C%ED%86%A0%EB%A6%AC%EC%96%BC](https://wannabe00.tistory.com/entry/Colab-%EC%B4%88%EA%B8%B0-%EC%84%A4%EC%A0%95-%ED%8A%9C%ED%86%A0%EB%A6%AC%EC%96%BC)  
22. 구글 코랩(colab)에서 구글 드라이브 폴더 접근하기 \- 삽질블로그 \- 티스토리, 6월 30, 2025에 액세스, [https://seonybob3210.tistory.com/11](https://seonybob3210.tistory.com/11)  
23. Colab에 구글 드라이브 csv파일 마운트하기 \- velog, 6월 30, 2025에 액세스, [https://velog.io/@kiache12/Colab%EC%97%90-%EA%B5%AC%EA%B8%80-%EB%93%9C%EB%9D%BC%EC%9D%B4%EB%B8%8C-csv%ED%8C%8C%EC%9D%BC-%EB%A7%88%EC%9A%B4%ED%8A%B8%ED%95%98%EA%B8%B0](https://velog.io/@kiache12/Colab%EC%97%90-%EA%B5%AC%EA%B8%80-%EB%93%9C%EB%9D%BC%EC%9D%B4%EB%B8%8C-csv%ED%8C%8C%EC%9D%BC-%EB%A7%88%EC%9A%B4%ED%8A%B8%ED%95%98%EA%B8%B0)  
24. 구글 코랩(Google Colab/Colaboratory) 사용법 \- release: canary \- 티스토리, 6월 30, 2025에 액세스, [https://canaryrelease.tistory.com/33](https://canaryrelease.tistory.com/33)  
25. \[파이썬\] 구글 코랩(Colab) 단축키 모음 \- Surf on Media \- 티스토리, 6월 30, 2025에 액세스, [https://surfonmedia.tistory.com/1](https://surfonmedia.tistory.com/1)  
26. 파이썬 데이터 사이언스 Cheat Sheet: NumPy 기초, 기본 \- taewan.kim 블로그, 6월 30, 2025에 액세스, [http://taewan.kim/post/numpy\_cheat\_sheet/](http://taewan.kim/post/numpy_cheat_sheet/)  
27. \[파이썬 개념\] Numpy \- 개념 총정리 (생성,추출,연산,통계,메소드,조건식,정렬 등) \- 오우진, 6월 30, 2025에 액세스, [https://oujin.tistory.com/entry/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EA%B0%9C%EB%85%90-Numpy-1-%EB%B0%B0%EC%97%B4-%EA%B0%9D%EC%B2%B4-ndarray](https://oujin.tistory.com/entry/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EA%B0%9C%EB%85%90-Numpy-1-%EB%B0%B0%EC%97%B4-%EA%B0%9D%EC%B2%B4-ndarray)  
28. 넘파이(NumPy) 기초: 배열 및 벡터 계산, 6월 30, 2025에 액세스, [https://compmath.korea.ac.kr/appmath/NumpyBasics.html](https://compmath.korea.ac.kr/appmath/NumpyBasics.html)  
29. \[Pandas\] Series와 DataFrame \- 준비하는 대학생 \- 티스토리, 6월 30, 2025에 액세스, [https://gsbang.tistory.com/entry/Pandas-Series%EC%99%80-DataFrame](https://gsbang.tistory.com/entry/Pandas-Series%EC%99%80-DataFrame)  
30. 5-4. pandas · 왕초보를 위한 파이썬 활용하기 \- cycorld, 6월 30, 2025에 액세스, [https://cycorld.gitbooks.io/python/content/pandas.html](https://cycorld.gitbooks.io/python/content/pandas.html)  
31. 파이썬 판다스 데이터프레임 예제 반드시 알아야 하는 함수들, python pandas Series and DataFrame \# 1 \- 존버력을 길러보자, 6월 30, 2025에 액세스, [https://koreadatascientist.tistory.com/96](https://koreadatascientist.tistory.com/96)  
32. 맷플롯립(Matplotlib), 데이터 시각화 알아보기 \- 괭이쟁이, 6월 30, 2025에 액세스, [https://laboputer.github.io/machine-learning/2020/05/04/matplitlib-tutorial/](https://laboputer.github.io/machine-learning/2020/05/04/matplitlib-tutorial/)  
33. \[Matplotlib\] Matplotlib을 이용해서 파이썬으로 그래프 그려보기, 6월 30, 2025에 액세스, [https://boringariel.tistory.com/22](https://boringariel.tistory.com/22)  
34. 파이썬을 이용한 데이타 시각화 \#1 \- Matplotlib 기본 그래프 그리기 \- 조대협의 블로그, 6월 30, 2025에 액세스, [https://bcho.tistory.com/1201](https://bcho.tistory.com/1201)
