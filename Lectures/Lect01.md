# 1주차: AI 프로그래밍 첫걸음 및 파이썬 기초 다지기 (Google Colab 환경)

## 주차별 학습 목표

이번 1주차 강의를 통해 여러분은 인공지능(AI)의 큰 그림을 이해하고, 파이썬 프로그래밍 기초를 다진 후, 파이토치(PyTorch)의 기본 개념을 익혀 데이터 처리의 첫 코드를 작성할 수 있게 됩니다. 또한, AI 기술과 함께 고려해야 할 윤리적 관점에 대해서도 알아보는 시간을 가집니다.

## 주요 학습 내용

### 1. 이론: AI, ML, DL의 개념 및 관계, AI 윤리

#### 1.1 인공지능(AI), 머신러닝(ML), 딥러닝(DL)의 개념 및 관계

* **인공지능 (Artificial Intelligence, AI)**
    * 인간의 지능을 모방하여 기계가 사고하고, 학습하며, 문제를 해결하는 기술 전반을 의미합니다.
    * 넓은 의미에서 인간처럼 생각하고 행동하는 모든 시스템을 포함합니다.
    * 예시: 자율 주행 자동차, 음성 비서 (Siri, Google Assistant), 추천 시스템 등

* **머신러닝 (Machine Learning, ML)**
    * AI의 하위 분야로, 명시적인 프로그래밍 없이 데이터로부터 학습하여 성능을 향상시키는 알고리즘과 기술을 의미합니다.
    * 데이터를 기반으로 패턴을 찾아 예측하거나 의사결정을 합니다.
    * **지도 학습 (Supervised Learning):** 정답(레이블)이 있는 데이터를 통해 학습합니다.
        * **회귀 (Regression):** 연속적인 값을 예측 (예: 집값 예측)
        * **분류 (Classification):** 범주형 값을 예측 (예: 스팸 메일 분류)
    * **비지도 학습 (Unsupervised Learning):** 정답(레이블)이 없는 데이터를 통해 학습합니다.
        * **군집화 (Clustering):** 데이터 간의 유사성을 기반으로 그룹화 (예: 고객 세분화)
        * **차원 축소 (Dimensionality Reduction):** 데이터의 특징 수를 줄임 (예: 이미지 압축)
    * **강화 학습 (Reinforcement Learning):** 에이전트가 특정 환경에서 보상을 최대화하는 방향으로 행동을 학습합니다. (예: 알파고, 로봇 제어)

* **딥러닝 (Deep Learning, DL)**
    * 머신러닝의 하위 분야로, 인공 신경망(Neural Networks)이라는 다층 구조를 활용하여 복잡한 패턴을 학습합니다.
    * 인간 뇌의 작동 방식을 모방하여 많은 양의 데이터를 통해 스스로 특징을 추출하고 학습합니다.
    * **주요 아키텍처:**
        * **CNN (Convolutional Neural Network):** 주로 이미지 처리 (예: 이미지 인식, 객체 탐지)
        * **RNN (Recurrent Neural Network):** 주로 시퀀스 데이터 처리 (예: 자연어 처리, 음성 인식)
        * **Transformer:** 최근 자연어 처리 분야에서 혁신을 가져온 아키텍처 (예: LLM, 기계 번역)

* **개념 관계 요약:**
    * AI는 가장 큰 개념이며, ML은 AI를 구현하는 한 방법입니다.
    * DL은 ML을 구현하는 한 방법이며, 특히 복잡한 문제 해결에 강력한 성능을 발휘합니다.
    * **AI ⊃ ML ⊃ DL**

**주요 AI, ML, DL 개념별 예시/사례 비교**

| 개념       | 하위 분야 (예시)       | 주요 사례/활용                                                                       |
| :--------- | :--------------------- | :----------------------------------------------------------------------------------- |
| **인공지능 (AI)** | -                      | - 자율 주행 자동차 (복합적인 지능 요구)                                              |
|            |                        | - 음성 비서 (Siri, Google Assistant 등 음성 인식 및 자연어 이해)                     |
|            |                        | - 추천 시스템 (사용자 선호도 학습 및 예측)                                           |
| **머신러닝 (ML)** | **지도 학습** | - **회귀:** 주택 가격 예측 (연속 값 예측)                                            |
|            |                        | - **분류:** 스팸 메일 분류 (정상/스팸), 암 진단 (양성/악성)                         |
|            | **비지도 학습** | - **군집화:** 고객 세분화 (유사한 고객 그룹화), 뉴스 기사 토픽 분류                   |
|            |                        | - **차원 축소:** 이미지 압축 (고차원 데이터를 저차원으로 표현), 빅데이터 시각화       |
|            | **강화 학습** | - **게임 플레이:** 알파고 (바둑), DeepMind의 StarCraft II AI                      |
|            |                        | - **로봇 제어:** 로봇 팔 움직임 최적화, 자율 주행 차량의 경로 계획 및 제어            |
| **딥러닝 (DL)** | **CNN** | - **이미지 인식:** 사진 속 객체 분류 (강아지/고양이), 얼굴 인식, 의료 영상 분석       |
|            |                        | - **객체 탐지:** 자율 주행차의 보행자, 차량, 신호등 위치 탐지                        |
|            | **RNN/LSTM** | - **음성 인식:** 음성-텍스트 변환 (음성 비서의 핵심 기술)                              |
|            |                        | - **기계 번역:** 구글 번역 (시퀀스 데이터를 다른 시퀀스로 변환)                       |
|            | **Transformer** | - **거대 언어 모델 (LLM):** ChatGPT, Gemini (텍스트 생성, 요약, 번역, 질문 응답)    |
|            |                        | - **이미지 생성:** DALL-E, Stable Diffusion (텍스트 설명으로 이미지 생성)            |


#### 1.2 AI의 역사 및 최근 기술 발전 동향

인공지능은 짧은 기간에 급격히 발전한 것처럼 보이지만, 그 뿌리는 수십 년 전으로 거슬러 올라갑니다. AI의 주요 발전 단계를 이해하는 것은 현재와 미래의 AI 기술을 조망하는 데 중요합니다.

* **초기 AI (1950년대 ~ 1970년대: 규칙 기반 시스템)**
    * **다트머스 회의 (1956년):** '인공지능(Artificial Intelligence)' 용어 최초 제안 (존 매카시 주도).
    * **논리 및 기호 기반 AI:**
        * 인간 지식을 규칙과 기호로 표현, 컴퓨터가 논리적 추론 및 문제 해결.
        * **ELIZA (1966년):** 사용자의 대화 패턴 인식, 간단한 응답 생성 (초기 자연어 처리).
        * **SHRDLU (1972년):** 특정 블록 세계에서 명령 이해 및 실행 (초기 자연어 이해).
    * **제한된 성공과 'AI 겨울':** 복잡한 현실 문제 해결 한계, 과도한 기대와 실망 반복.

* **머신러닝의 부상 (1980년대 ~ 2000년대: 통계적 접근)**
    * **전문가 시스템:** 특정 분야 전문가 지식 내재화, 진단/상담 수행 (예: MYCIN).
    * **인공 신경망 재조명:**
        * 퍼셉트론(Perceptron) 한계 지적.
        * 역전파(Backpropagation) 알고리즘 발전으로 다층 신경망 학습 가능성 열림.
    * **통계적 머신러닝 발전:**
        * Support Vector Machine (SVM), 의사결정 트리(Decision Tree), 은닉 마르코프 모델(HMM) 등 통계적 모델 발전.
        * 데이터 기반 패턴 학습 및 예측 능력 향상.

* **딥러닝 혁명과 AI의 황금기 (2010년대 ~ 현재: 빅데이터와 GPU)**
    * **데이터 증가:** 인터넷 및 센서 기술 확산으로 빅데이터 생성.
    * **컴퓨팅 파워 증가 (GPU 등장):**
        * NVIDIA CUDA 플랫폼 및 GPU 발전, 딥러닝 모델 병렬 연산 가속화.
        * 수십억 개 연산 동시 처리 물리적 장벽 해소.
    * **알고리즘 발전:**
        * **CNN (Convolutional Neural Network):** 이미지 인식 대회(ImageNet)에서 기존 기술 압도, 딥러닝 시대 개막 (AlexNet, VGG, ResNet 등).
        * **RNN/LSTM (Recurrent Neural Network / Long Short-Term Memory):** 시퀀스 데이터 처리(텍스트, 음성) 탁월, 자연어 처리 및 음성 인식 혁신.
        * **Transformer (2017년):** 구글 발표, Self-Attention 메커니즘으로 RNN 한계 극복, 자연어 처리 표준 (LLM 기반).
    * **전이 학습 (Transfer Learning):** 대규모 사전 학습 모델(Pre-trained model) 미세 조정(Fine-tuning) 기법 보편화, 데이터 부족 분야 딥러닝 활용 가능.

* **최근 기술 발전 동향 (2020년대 ~ 현재)**
    * **거대 언어 모델 (Large Language Models, LLM):**
        * **설명:** 대규모 텍스트 데이터셋 학습, 인간 언어 이해 및 생성 딥러닝 모델. Transformer 아키텍처 기반, 수십억~수천억 개 파라미터 보유.
        * **주요 특징:** 방대한 지식 학습, 높은 유연성, Few-shot/Zero-shot 학습 능력.
        * **주요 플레이어 및 사례:**
            * **OpenAI:** GPT 시리즈 (GPT-3, GPT-3.5, GPT-4, GPT-4o) - ChatGPT로 대중화.
            * **Google:** PaLM 2, Gemini - 멀티모달 기능 겸비, 복합 추론 능력 강점.
            * **Meta:** LLaMA, LLaMA 2, LLaMA 3 - 오픈 소스 기반, 연구 커뮤니티에 영향.
            * **Anthropic:** Claude, Claude 2, Claude 3 - 안전하고 유용한 AI 목표.
            * **Naver, Kakao (한국):** HyperCLOVA X, KoGPT - 한국어 특화 LLM 개발.
    * **생성형 AI (Generative AI):**
        * **설명:** 기존 데이터 패턴 학습 후 새로운 원본 콘텐츠(텍스트, 이미지, 오디오, 비디오 등) 생성 기술. VAE, GAN, Diffusion Model 등 아키텍처 활용.
        * **주요 특징:** 창의적 콘텐츠 생성, 데이터 기반 사실적 결과물, 다양한 미디어 형식 지원.
        * **주요 플레이어 및 사례:**
            * **OpenAI:** DALL-E (텍스트-이미지), Sora (텍스트-비디오).
            * **Google:** Imagen (텍스트-이미지), MusicLM (텍스트-음악), Veo (텍스트-비디오).
            * **Midjourney:** 고품질 이미지 생성 전문 서비스.
            * **Stability AI:** Stable Diffusion (오픈 소스 텍스트-이미지 생성).
            * **RunwayML:** 텍스트-비디오, 이미지-비디오 등 비디오 생성/편집 도구.
    * **멀티모달 AI (Multimodal AI):**
        * **설명:** 두 가지 이상 다른 유형의 데이터(텍스트, 이미지, 오디오, 비디오 등)를 동시에 이해하고 상호작용하는 AI 모델.
        * **주요 특징:** 다양한 정보원 통합 이해, 복합적 추론 및 생성 능력, 현실 세계 복잡성 반영.
        * **주요 플레이어 및 사례:**
            * **Google:** Gemini (텍스트, 이미지, 오디오, 비디오), PaLM-E (언어-로봇).
            * **OpenAI:** GPT-4V (텍스트+이미지), CLIP (텍스트-이미지 매칭).
            * **Meta:** ImageBind (6가지 양식의 데이터 연결).
    * **온디바이스 AI (On-Device AI) / 엣지 AI (Edge AI):**
        * **설명:** 클라우드 서버 아닌 스마트폰, 자율주행차, IoT 기기 등 최종 사용자 기기에서 AI 모델 추론(Inference) 연산 직접 수행.
        * **주요 특징:** 낮은 지연 시간, 강화된 개인 정보 보호, 네트워크 의존성 감소, 전력 효율성 중요.
        * **주요 플레이어 및 사례:**
            * **Apple:** Neural Engine (iPhone ML 가속기), Siri.
            * **Google:** Google Pixel Tensor 칩 (온디바이스 AI 연산 가속), Google Assistant 온디바이스 음성 인식.
            * **Qualcomm:** Snapdragon Processors NPU (스마트폰 AI 연산 가속).
            * **Tesla:** 자율주행 차량 자체 AI 칩 (FSD Chip).
            * **각종 IoT 기기:** 스마트 카메라 자체 객체 인식, 스마트 스피커 로컬 음성 명령 처리.

이러한 플레이어들은 각자의 강점을 바탕으로 AI 기술의 발전과 확산에 기여하고 있습니다.

#### 1.3 AI 분야별 주요 활용 및 비교

AI 기술은 다양한 분야에 적용되어 혁신을 이끌고 있으며, 각 분야마다 특화된 기술과 활용 사례를 가집니다.

| 분야           | 주요 기술/알고리즘                 | 주요 활용 사례                                                                                                   |
| :------------- | :------------------------------- | :--------------------------------------------------------------------------------------------------------------- |
| **컴퓨터 비전** | CNN (ResNet, VGG, YOLO 등), Transformer (ViT) | 이미지 분류 (고양이/개), 객체 탐지 (자율주행차의 보행자 인식), 안면 인식, 이미지 생성 (DALL-E), 의료 영상 분석          |
| **자연어 처리 (NLP)** | RNN/LSTM, Transformer (BERT, GPT, T5 등), 통계적 언어 모델 | 기계 번역 (Google 번역), 챗봇 및 가상 비서, 감성 분석, 텍스트 요약, 질문 응답, 거대 언어 모델(LLM) 기반 콘텐츠 생성 |
| **음성 인식** | RNN/LSTM, Transformer, Hidden Markov Model (HMM) | 음성-텍스트 변환 (음성 비서), 음성 명령 제어 (스마트 스피커), 의료 녹취록 자동 생성, 화자 인식                       |
| **로봇 공학** | 강화 학습, 컴퓨터 비전, 제어 이론, SLAM | 자율 이동 로봇 (물류 창고), 휴머노이드 로봇 (서비스), 산업용 로봇 (생산 자동화), 로봇 팔 제어, 드론 제어                |
| **헬스케어** | 딥러닝 (CNN), 머신러닝 (SVM, 랜덤 포레스트), 데이터 마이닝 | 질병 진단 보조 (의료 영상 판독), 신약 개발 (화합물 예측), 개인 맞춤형 치료법 제안, 의료 기록 분석 및 예측           |
| **금융** | 시계열 분석 (RNN), 머신러닝 (회귀, 분류), 강화 학습 | 주가 예측, 신용 평가, 사기 탐지 (카드 사기), 알고리즘 트레이딩, 챗봇 기반 금융 상담                               |
| **추천 시스템** | 협업 필터링, 콘텐츠 기반 필터링, 딥러닝 (Factorization Machines) | 온라인 쇼핑몰 상품 추천, 스트리밍 서비스 영화/음악 추천, 소셜 미디어 콘텐츠 추천                                 |
| **자율 주행** | 컴퓨터 비전 (객체 탐지, 분할), 강화 학습, 센서 융합 | 차선 유지, 교통 신호 인식, 보행자 및 장애물 회피, 경로 계획, 센서 데이터 처리 및 상황 인지                         |

#### 1.4 주요 AI 발전 동향 분야별 비교 (2020년대 이후)

| 분야           | 핵심 개념                                                  | 주요 특징                                                                                               | 주요 플레이어/연구기관              | 대표 사례/기술                                                         |
| :------------- | :------------------------------------------------------- | :------------------------------------------------------------------------------------------------------ | :---------------------------------- | :--------------------------------------------------------------------- |
| **거대 언어 모델 (LLM)** | 방대한 텍스트 데이터 학습, 인간 언어 이해 및 생성             | - 대규모 파라미터 (수십억~수천억)<br>- 복잡한 문맥 이해 및 생성 능력<br>- 다양한 언어 작업 수행 가능 (요약, 번역, 추론) | OpenAI, Google, Meta, Anthropic, Naver, Kakao | GPT-4, Gemini, LLaMA, Claude, HyperCLOVA X, Bard                      |
| **생성형 AI** | 기존 데이터 패턴 학습 후 새로운 원본 콘텐츠 생성              | - 텍스트, 이미지, 오디오, 비디오 등 다양한 형식 지원<br>- 창의적이고 사실적인 결과물 생성               | OpenAI, Google, Midjourney, Stability AI, RunwayML | DALL-E, Imagen, Stable Diffusion, Sora, MusicLM, Midjourney            |
| **멀티모달 AI** | 텍스트, 이미지, 오디오 등 여러 종류 데이터 동시 이해 및 상호작용 | - 복합적인 정보원 통합 및 관계 학습<br>- 현실 세계의 복잡한 상황 이해 및 추론<br>- 다양한 형식의 입력/출력 | Google, OpenAI, Meta                | Gemini, GPT-4V, CLIP, ImageBind                                        |
| **온디바이스 AI** | 클라우드 없이 기기 자체에서 AI 추론 수행                   | - 낮은 지연 시간, 즉각적인 반응<br>- 강화된 개인 정보 보호<br>- 네트워크 의존성 감소, 전력 효율성 중요   | Apple, Google, Qualcomm, Tesla      | iPhone Neural Engine, Pixel Tensor 칩, Snapdragon NPU, Tesla FSD Chip |

#### 1.5 AI 윤리 및 사회적 영향 (편향성, 공정성 등 간단한 논의)

AI 기술의 발전은 사회에 긍정적인 영향을 미치지만, 동시에 새로운 윤리적, 사회적 문제를 야기할 수 있습니다.

* **편향성 (Bias):**
    * AI 모델이 학습 데이터에 포함된 편향을 그대로 학습하여 특정 그룹에 불이익을 주거나 차별적인 결과를 내놓을 수 있습니다.
    * 예시: 성별, 인종, 지역 등에 대한 편견이 학습 데이터에 반영되어 모델의 의사결정에 영향을 미치는 경우.
* **공정성 (Fairness):**
    * AI 시스템이 모든 사람에게 공정하고 차별 없이 작동해야 한다는 원칙입니다.
    * 편향성을 줄이고 모든 사용자에게 균등한 기회와 결과를 제공하기 위한 노력이 필요합니다.
* **책임 (Accountability):**
    * AI 시스템의 오작동이나 잘못된 결정으로 인해 발생하는 문제에 대한 책임 소재를 명확히 해야 합니다.
* **투명성 (Transparency):**
    * AI 모델의 의사결정 과정을 이해하고 설명할 수 있도록 해야 합니다. '블랙박스' 문제 해결 노력.
* **개인 정보 보호 (Privacy):**
    * AI 학습에 사용되는 대규모 데이터셋은 개인 정보를 포함할 수 있으며, 이에 대한 철저한 보호가 필요합니다.

**논의의 중요성:** AI 개발자로서 기술적 능력뿐만 아니라 사회적 영향과 윤리적 책임에 대한 인식을 갖는 것이 중요합니다.

---

### 2. 실습: Google Colab, Python, PyTorch 기본

#### 2.1 Google Colab 개발 환경 설정 및 사용법 익히기

Google Colaboratory (Colab)는 Google에서 제공하는 클라우드 기반의 Jupyter Notebook 환경입니다. 별도의 설치 없이 웹 브라우저에서 Python 코드를 실행하고 GPU/TPU를 무료로 사용할 수 있어 AI 학습에 매우 유용합니다.

* **Google Colab 접속:**
    * 웹 브라우저에서 [colab.research.google.com](https://colab.research.google.com/) 에 접속합니다.
    * Google 계정으로 로그인합니다.

* **새 노트북 생성:**
    * Colab 시작 화면에서 '새 노트북'을 클릭하거나, '파일' > '새 노트북'을 선택하여 새 `.ipynb` 파일을 생성합니다.

* **코드/텍스트 셀 활용:**
    * **코드 셀:** Python 코드를 작성하고 실행하는 공간입니다. 셀을 클릭하고 코드를 입력한 후 `Shift + Enter`를 누르거나, 셀 좌측의 실행 버튼(▶)을 클릭하여 실행합니다.
    * **텍스트 셀:** Markdown 문법을 사용하여 설명을 작성하는 공간입니다. 셀을 클릭하고 텍스트를 입력한 후 `Shift + Enter`를 누르거나, 셀 외부를 클릭하여 렌더링된 텍스트를 확인합니다. Markdown 문법은 텍스트 서식, 제목, 목록, 이미지 등을 쉽게 추가할 수 있게 해줍니다.

* **런타임 유형 변경 (GPU/TPU 사용 설정):**
    * AI 모델 학습 시 GPU나 TPU를 사용하면 계산 속도를 크게 높일 수 있습니다.
    * '런타임' > '런타임 유형 변경' 메뉴로 이동합니다.
    * '하드웨어 가속기' 드롭다운 메뉴에서 'GPU' 또는 'TPU'를 선택한 후 '저장'을 클릭합니다.
    * **실습:** GPU 런타임으로 변경 후 `!nvidia-smi` 명령어를 코드 셀에 입력하여 GPU가 할당되었는지 확인해 보세요. (Colab GPU 런타임 설정 방법 참고)

* **Colab 파일/Google Drive 연동:**
    * Colab 노트북은 Google Drive에 자동으로 저장됩니다.
    * Google Drive의 파일을 Colab에서 불러오거나, Colab에서 생성된 파일을 Google Drive에 저장할 수 있습니다.
    * Google Drive 마운트 코드 (코드 셀에 입력 후 실행):
        ```python
        from google.colab import drive
        drive.mount('/content/drive')
        ```
        이 코드를 실행하면 인증 절차를 거쳐 Google Drive를 Colab 환경에 마운트할 수 있습니다. 마운트 후에는 `/content/drive/MyDrive/` 경로를 통해 Google Drive 파일에 접근할 수 있습니다.

#### 2.2 Python 기본 문법 및 자료 구조

파이썬은 AI 프로그래밍의 핵심 언어입니다. 다음 기본 문법과 자료 구조를 Colab 코드 셀에서 직접 실습해 봅시다.

* **변수와 자료형:**
    * 변수는 데이터를 저장하는 공간입니다. 파이썬은 변수 선언 시 자료형을 명시하지 않아도 됩니다.
    * **정수 (int):** `num = 10`
    * **실수 (float):** `pi = 3.14`
    * **문자열 (str):** `name = "Alice"`
    * **논리형 (bool):** `is_student = True`
    * ```python
        # 변수 선언 및 할당
        integer_var = 10
        float_var = 3.14
        string_var = "Hello, PyTorch!"
        boolean_var = True

        # 자료형 확인
        print(f"integer_var: {integer_var}, type: {type(integer_var)}")
        print(f"float_var: {float_var}, type: {type(float_var)}")
        print(f"string_var: {string_var}, type: {type(string_var)}")
        print(f"boolean_var: {boolean_var}, type: {type(boolean_var)}")
        ```

* **연산자:**
    * 산술 연산자 (`+`, `-`, `*`, `/`, `%`, `**`, `//`)
    * 비교 연산자 (`==`, `!=`, `<`, `>`, `<=`, `>=`)
    * 논리 연산자 (`and`, `or`, `not`)
    * ```python
        # 산술 연산
        a = 10
        b = 3
        print(f"a + b = {a + b}")
        print(f"a / b = {a / b}") # 나눗셈 (실수)
        print(f"a // b = {a // b}") # 몫 (정수)
        print(f"a % b = {a % b}") # 나머지
        print(f"a ** b = {a ** b}") # 거듭제곱

        # 비교 연산
        x = 5
        y = 10
        print(f"x == y: {x == y}")
        print(f"x < y: {x < y}")

        # 논리 연산
        p = True
        q = False
        print(f"p and q: {p and q}")
        print(f"p or q: {p or q}")
        print(f"not p: {not p}")
        ```

* **조건문 (if, elif, else):**
    * 특정 조건에 따라 다른 코드를 실행합니다.
    * ```python
        score = 85
        if score >= 90:
            print("A 학점")
        elif score >= 80:
            print("B 학점")
        else:
            print("C 학점 이하")
        ```

* **반복문 (for, while):**
    * **for 문:** 특정 횟수만큼 또는 시퀀스의 각 요소에 대해 반복합니다.
        * ```python
            # for 문 예시 (리스트 반복)
            fruits = ["apple", "banana", "cherry"]
            for fruit in fruits:
                print(fruit)

            # for 문 예시 (범위 반복)
            for i in range(5): # 0부터 4까지
                print(i)
            ```
    * **while 문:** 조건이 참인 동안 반복합니다.
        * ```python
            # while 문 예시
            count = 0
            while count < 3:
                print(f"Count: {count}")
                count += 1
            ```

* **함수 (def):**
    * 재사용 가능한 코드 블록을 정의합니다.
    * ```python
        # 함수 정의
        def greet(name):
            return f"Hello, {name}!"

        # 함수 호출
        message = greet("PyAI Learner")
        print(message)

        def add_numbers(x, y):
            return x + y

        result = add_numbers(7, 3)
        print(f"7 + 3 = {result}")
        ```

* **파이썬 자료 구조:**

    * **리스트 (List):** 순서가 있고 변경 가능한(mutable) 데이터의 집합입니다. `[]`로 표현합니다.
        * ```python
            my_list = [1, 2, 3, "four", True]
            print(f"리스트: {my_list}")
            print(f"첫 번째 요소: {my_list[0]}")
            my_list.append(5) # 요소 추가
            print(f"요소 추가 후: {my_list}")
            my_list[0] = 10 # 요소 변경
            print(f"첫 번째 요소 변경 후: {my_list}")
            ```

    * **튜플 (Tuple):** 순서가 있고 변경 불가능한(immutable) 데이터의 집합입니다. `()`로 표현합니다.
        * ```python
            my_tuple = (1, 2, "three", False)
            print(f"튜플: {my_tuple}")
            print(f"두 번째 요소: {my_tuple[1]}")
            # my_tuple.append(4) # 에러 발생: 튜플은 변경 불가능
            ```

    * **딕셔너리 (Dictionary):** 키-값(key-value) 쌍으로 이루어진 변경 가능한(mutable) 데이터의 집합입니다. `{}`로 표현하며, 키는 중복될 수 없고 값은 중복될 수 있습니다.
        * ```python
            my_dict = {"name": "Alice", "age": 30, "city": "Seoul"}
            print(f"딕셔너리: {my_dict}")
            print(f"이름: {my_dict['name']}")
            my_dict["age"] = 31 # 값 변경
            my_dict["job"] = "Engineer" # 새로운 키-값 추가
            print(f"변경 및 추가 후: {my_dict}")
            ```

    * **세트 (Set):** 중복을 허용하지 않으며, 순서가 없는(unordered) 데이터의 집합입니다. `{}` 또는 `set()`으로 표현합니다.
        * ```python
            my_set = {1, 2, 3, 2, 1} # 중복 제거
            print(f"세트: {my_set}")
            my_set.add(4)
            print(f"요소 추가 후: {my_set}")
            ```

#### 2.3 PyTorch 기본: Tensor의 개념, 생성 및 기본 조작

PyTorch는 딥러닝 모델을 구축하고 학습시키는 데 사용되는 오픈소스 머신러닝 라이브러리입니다. PyTorch의 핵심은 `Tensor`입니다.

* **Tensor (텐서) 개념:**
    * Numpy의 `ndarray`와 유사한 다차원 배열.
    * GPU에서 연산을 가속화 가능 (주요 차이점).
    * 딥러닝 모델의 모든 입력/출력 데이터, 가중치, 편향은 텐서로 표현.

* **Colab에서 PyTorch 설치 및 확인:**
    * Colab에는 PyTorch 기본 설치되어 있음. 다음 코드로 설치 여부 및 버전 확인:
    * ```python
        # PyTorch 설치 확인 (Colab에는 기본 설치되어 있음)
        # !pip install torch torchvision torchaudio

        import torch
        print(f"PyTorch 버전: {torch.__version__}")

        # GPU 사용 가능 여부 확인
        if torch.cuda.is_available():
            print(f"GPU 사용 가능: {torch.cuda.get_device_name(0)}")
        else:
            print("GPU 사용 불가능")
        ```

* **Tensor 생성:**
    * 다양한 방법으로 텐서 생성 가능:
        * `torch.empty(shape)`: 초기화되지 않은 텐서.
        * `torch.zeros(shape)`: 모든 요소가 0인 텐서.
        * `torch.ones(shape)`: 모든 요소가 1인 텐서.
        * `torch.rand(shape)`: 0과 1 사이 균일 분포 난수 텐서.
        * `torch.randn(shape)`: 표준 정규 분포 난수 텐서.
        * `torch.tensor(data)`: Python 리스트나 Numpy 배열로부터 생성.
        * ```python
            # 초기화되지 않은 텐서
            x = torch.empty(5, 3) # 5x3 행렬
            print(f"Empty Tensor:\n{x}\n")

            # 0으로 채워진 텐서
            zeros_tensor = torch.zeros(3, 4)
            print(f"Zeros Tensor:\n{zeros_tensor}\n")

            # 1로 채워진 텐서
            ones_tensor = torch.ones(2, 2)
            print(f"Ones Tensor:\n{ones_tensor}\n")

            # 무작위 텐서
            rand_tensor = torch.rand(2, 3)
            print(f"Random Tensor:\n{rand_tensor}\n")

            # Python 리스트로부터 텐서 생성
            data = [[1, 2], [3, 4]]
            list_tensor = torch.tensor(data)
            print(f"From List:\n{list_tensor}\n")

            # Numpy 배열로부터 텐서 생성
            import numpy as np
            numpy_array = np.array([[5, 6], [7, 8]])
            numpy_tensor = torch.tensor(numpy_array)
            print(f"From NumPy Array:\n{numpy_tensor}\n")
            ```

* **Tensor 속성 (Attributes):**
    * `shape`: 텐서의 크기 (각 차원의 크기).
    * `dtype`: 텐서에 저장된 데이터의 자료형.
    * `device`: 텐서가 저장된 장치 (CPU 또는 GPU).
    * ```python
        tensor = torch.rand(3, 4)
        print(f"Tensor:\n{tensor}\n")
        print(f"Shape: {tensor.shape}")
        print(f"Data type: {tensor.dtype}")
        print(f"Device: {tensor.device}")
        ```

* **Tensor 조작 (Operations):**
    * **덧셈:**
        * `tensor1 + tensor2`
        * `torch.add(tensor1, tensor2)`
        * `tensor1.add_(tensor2)` (인플레이스 연산: `tensor1` 자체가 변경됨).
        * ```python
            t1 = torch.tensor([[1., 2.], [3., 4.]])
            t2 = torch.tensor([[5., 6.], [7., 8.]])

            print(f"t1 + t2:\n{t1 + t2}\n")
            print(f"torch.add(t1, t2):\n{torch.add(t1, t2)}\n")

            t1.add_(t2) # 인플레이스 연산: t1의 값이 바뀜
            print(f"t1 after add_:\n{t1}\n")
            ```
    * **곱셈:**
        * 요소별 곱셈: `tensor1 * tensor2` 또는 `torch.mul(tensor1, tensor2)`.
        * 행렬 곱셈: `tensor1 @ tensor2` 또는 `torch.matmul(tensor1, tensor2)`.
        * ```python
            t_a = torch.tensor([[1, 2], [3, 4]])
            t_b = torch.tensor([[5, 6], [7, 8]])

            # 요소별 곱셈
            print(f"Element-wise multiplication:\n{t_a * t_b}\n")

            # 행렬 곱셈
            print(f"Matrix multiplication:\n{t_a @ t_b}\n")
            ```
    * **인덱싱 및 슬라이싱:** Numpy와 유사하게 텐서의 특정 부분에 접근.
        * ```python
            tensor_example = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            print(f"Original Tensor:\n{tensor_example}\n")

            # 첫 번째 행
            print(f"First row: {tensor_example[0]}\n")

            # 두 번째 열
            print(f"Second column: {tensor_example[:, 1]}\n")

            # 부분 텐서 (1행부터 2행까지, 0열부터 1열까지)
            print(f"Slice: {tensor_example[1:3, 0:2]}\n")
            ```
    * **형태 변경 (Reshape):** `view()` 또는 `reshape()` 메서드를 사용하여 텐서 차원 형태 변경.
        * ```python
            x = torch.randn(4, 4)
            print(f"Original shape: {x.shape}")

            y = x.view(16) # 1차원 텐서로 변경
            print(f"Reshaped to 1D: {y.shape}")

            z = x.view(2, 8) # 2x8 텐서로 변경
            print(f"Reshaped to 2x8: {z.shape}")

            # -1은 PyTorch가 자동으로 크기를 추론하도록 합니다.
            a = x.view(-1, 8) # 8열로, 행은 자동으로
            print(f"Reshaped with -1: {a.shape}")
            ```

* **Tensor를 CPU와 GPU 간 이동:**
    * `tensor.to('cuda')`: 텐서를 GPU로 이동 (GPU 사용 가능할 경우).
    * `tensor.to('cpu')`: 텐서를 CPU로 이동.
    * ```python
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_tensor = torch.ones(5, 5, device=device) # 생성 시점에 GPU에 할당
            print(f"GPU Tensor:\n{gpu_tensor}\nDevice: {gpu_tensor.device}\n")

            cpu_tensor = gpu_tensor.to("cpu") # GPU 텐서를 CPU로 이동
            print(f"CPU Tensor (moved from GPU):\n{cpu_tensor}\nDevice: {cpu_tensor.device}\n")
        else:
            print("GPU를 사용할 수 없어 텐서 이동 실습을 건너_ㅂ니다.")
        ```

#### 2.4 GPU (Graphics Processing Unit) 심층 이해 및 AI 활용

GPU는 그래픽 처리 장치로, 본래 컴퓨터 그래픽 렌더링에 특화된 하드웨어입니다. 하지만 병렬 처리 능력 때문에 AI, 특히 딥러닝 분야에서 CPU를 뛰어넘는 핵심적인 역할을 수행합니다.

##### 2.4.1 GPU란 무엇인가?

* CPU(Central Processing Unit)는 '만능 일꾼'이라면, GPU(Graphics Processing Unit)는 '병렬 처리 전문가'.
* **CPU와의 비교:**
    * **CPU:** 소수의 강력하고 복잡한 코어(예: 4~16개), 복잡한 명령어 순차적/빠르게 처리 능함. 운영체제 실행, 문서 편집, 웹 브라우징 등 다양한 작업 적합. 파이프라인 최적화, 분기 예측 등 복잡한 제어 로직 포함.
    * **GPU:** 수천 개의 작고 단순한 코어(예: 수백~수천 개), 동일 연산 동시 수행에 최적화. 그래픽 렌더링 시 수많은 픽셀 색상 동시 계산 등 병렬성 높은 작업에 매우 효율적. 복잡한 제어 로직보다 연산 유닛 밀도 높이는 데 중점.
* **아키텍처적 특성:**
    * 그래픽 처리 파이프라인에 맞춰 설계, 대규모 데이터 행렬/벡터 연산 등 병렬 수학 연산 효율적 처리.
    * 딥러닝 모델 학습 과정은 대규모 선형대수 연산 반복, GPU 병렬 처리 능력과 완벽 일치.
    * CPU 대비 훨씬 높은 메모리 대역폭, 대량 데이터 빠른 읽기/쓰기 가능, 딥러닝 모델 거대 데이터셋 처리 시 병목 현상 감소.

##### 2.4.2 CUDA (Compute Unified Device Architecture)

* **정의 및 목적:**
    * NVIDIA 개발 병렬 컴퓨팅 플랫폼이자 API 모델.
    * NVIDIA GPU 병렬 처리 능력 활용, 일반 계산 작업 수행 가능하게 하는 핵심 기술.
    * GPU를 그래픽 처리뿐 아니라 범용 병렬 컴퓨팅(GPGPU) 수행 소프트웨어 계층.
    * C, C++, Fortran 등 익숙한 언어로 GPU 코드 작성 가능, 프로그래밍 접근성 높임.
    * 핵심 목적: GPU 방대한 코어 효율적 프로그래밍, 대규모 병렬 문제 해결 가속화.
* **CUDA의 역할 및 작동 방식:**
    * **하드웨어 추상화:** GPU 하드웨어 복잡성 추상화, 개발자 고수준 프로그래밍 모델 통해 GPU 접근 용이.
    * **커널(Kernel):** CUDA 프로그래밍 핵심 개념. GPU 수많은 스레드(Thread)들이 병렬 실행하는 함수. (예: 딥러닝 행렬 곱셈 시 각 요소 곱셈은 개별 스레드 동시 처리 커널 연산).
    * **메모리 관리:** GPU 내부 메모리(글로벌, 셰어드, 레지스터 등) 효율적 관리, CPU-GPU 데이터 전송 최적화 API 제공. (딥러닝 텐서 이동 및 결과 반환 시 중요).
    * **딥러닝 프레임워크 연동:** PyTorch/TensorFlow 등 딥러닝 프레임워크 내부적으로 CUDA 활용, GPU 병렬 연산 효율적 제어. (예: `tensor.to('cuda')` 시 CUDA API 통해 GPU 메모리 복사, 연산 실행).

##### 2.4.3 AI에서의 GPU 활용 현황

딥러닝 모델 학습은 대규모 행렬 곱셈, 벡터 덧셈, 합성곱 연산 등 병렬성 높은 선형대수 연산의 반복. GPU 환경에서 압도적 효율성.

* **학습 시간의 혁신적 단축:**
    * 수백만~수십억 개 파라미터 대규모 딥러닝 모델 학습에 막대한 계산량 요구.
    * CPU만으로는 며칠/몇 주/몇 달 소요될 작업, GPU 사용 시 몇 시간/몇 분 내 완료 가능.
    * 복잡한 CNN, RNN/LSTM, Transformer 등 심층 신경망 모델에서 효과 두드러짐.
* **대규모 모델 및 데이터셋 처리 능력:**
    * 고해상도 이미지, 방대한 텍스트 코퍼스 등 AI 모델 데이터셋 크기 지속 증가.
    * GPU의 높은 연산 능력 및 CPU 대비 월등한 메모리 대역폭은 대규모 데이터 효율적 로드/처리 필수.
    * 최근 LLM, 확산 모델은 수십억 파라미터 보유, 단일 GPU 부족 시 분산 학습 환경 필요.
* **AI 연구 및 개발의 가속화:**
    * GPU 덕분에 새로운 모델 아키텍처 빠른 실험, 하이퍼파라미터 튜닝 효율적 수행.
    * 모델 성능 최적화, 신기술 개발 및 상용화에 결정적 역할.
    * 실시간 객체 탐지, 강화 학습, 고품질 이미지/비디오 생성 등 GPU 없이는 어려웠던 AI 애플리케이션 등장.

##### 2.4.4 Google Colab 환경에서 GPU 성능 활용 방법

Google Colab은 별도 고성능 하드웨어 구매 없이 클라우드 기반 GPU 자원(무료/유료) 제공, 딥러닝 학습 수행 가능.

* **런타임 유형 변경을 통한 GPU 할당:**
    * **설정 경로:** Colab 노트북 상단 메뉴 `런타임(Runtime)` -> `런타임 유형 변경(Change runtime type)`.
    * **하드웨어 가속기 설정:** '하드웨어 가속기(Hardware accelerator)' 드롭다운 메뉴에서 `GPU` 또는 `TPU` 선택.
        * `GPU`: NVIDIA GPU, 대부분 딥러닝 작업에 범용 사용.
        * `TPU`: Google 개발 ASIC, TensorFlow/JAX 기반 대규모 모델 학습 최적화.
    * **저장:** 선택 후 '저장' 클릭, 새 런타임(세션) 할당.
* **할당된 GPU 정보 확인 코드:**
    * 런타임 유형 `GPU` 변경 후, 다음 코드를 Colab 코드 셀에 입력/실행하여 GPU 상세 정보 확인.
    * `!nvidia-smi` 명령어: Linux 시스템에서 NVIDIA GPU 상태 표시 (모델명, 드라이버, CUDA 버전, 사용량, 메모리 등).
    * ```python
        import torch

        # CUDA (GPU) 사용 가능 여부 확인
        if torch.cuda.is_available():
            print("CUDA (GPU) 사용 가능!")
            print(f"할당된 GPU: {torch.cuda.get_device_name(0)}") # 할당된 첫 번째 GPU 이름
            print(f"GPU 개수: {torch.cuda.device_count()}")
            # NVIDIA GPU 정보 출력 (리눅스 명령어)
            !nvidia-smi
        else:
            print("CUDA (GPU) 사용 불가능. 런타임 유형을 'GPU'로 설정했는지 확인하세요.")

        # PyTorch에서 사용할 장치 설정 (GPU가 없으면 CPU)
        # 이 변수를 사용하여 모델과 텐서를 해당 장치로 옮깁니다.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"현재 텐서 연산에 사용될 장치: {device}")
        ```
* **텐서(Tensor)를 GPU로 이동시키기:**
    * PyTorch는 텐서와 모델을 기본적으로 CPU 메모리에서 생성. GPU 이점 활용 위해 명시적으로 GPU 메모리로 이동 필요.
    * `tensor.to(device)` 또는 `model.to(device)` 메서드 사용, 텐서/모델을 GPU로 이동.
    * **주의:** CPU/GPU 텐서는 서로 다른 메모리 공간, 직접 연산 불가. 연산 수행 시 동일 장치에 있어야 함.
    * **예시:**
        ```python
        import torch

        # 사용할 장치 설정 (GPU가 없으면 CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"사용할 장치: {device}")

        # CPU에 생성된 텐서
        cpu_tensor = torch.randn(3, 3)
        print(f"\nCPU 텐서:\n{cpu_tensor}\nDevice: {cpu_tensor.device}")

        # 텐서를 GPU로 이동 (GPU 사용 가능 시)
        if device.type == 'cuda':
            gpu_tensor = cpu_tensor.to(device)
            print(f"\nGPU 텐서 (이동 후):\n{gpu_tensor}\nDevice: {gpu_tensor.device}")

            # GPU에서 연산 수행
            result_tensor = gpu_tensor * 2 + 10
            print(f"\nGPU 연산 결과 텐서:\n{result_tensor}\nDevice: {result_tensor.device}")

            # 다시 CPU로 이동 (결과를 CPU에서 사용해야 할 경우)
            result_on_cpu = result_tensor.to('cpu')
            print(f"\nCPU로 다시 이동된 결과 텐서:\n{result_on_cpu}\nDevice: {result_on_cpu.device}")
        else:
            print("\nGPU를 사용할 수 없어 텐서 이동 및 GPU 연산 실습을 건너_ㅂ니다.")
            # GPU가 없을 때는 CPU에서 연산
            result_tensor_cpu = cpu_tensor * 2 + 10
            print(f"\nCPU 연산 결과 텐서:\n{result_tensor_cpu}\nDevice: {result_tensor_cpu.device}")
        ```

##### 2.4.5 GPU를 활용한 처리 과정

AI 학습 시 GPU를 활용하는 일반적인 처리 과정은 데이터 이동, 연산 수행, 결과 저장의 효율성을 극대화합니다.

1.  **데이터 준비 및 로드 (CPU):**
    * 원본 데이터셋 하드 디스크에서 읽어와 CPU 메인 메모리에 로드.
    * 데이터 전처리(정규화, 크기 조정, 인코딩 등) 수행 가능.
2.  **모델 정의 (CPU 메모리 초기화):**
    * PyTorch의 `nn.Module` 등으로 딥러닝 모델 구조 정의.
    * 모델 초기 가중치/편향은 기본적으로 CPU 메모리에 생성.
3.  **모델 및 데이터 GPU 이동 (CPU -> GPU):**
    * **모델 이동:** `model.to('cuda')` 또는 `model.to(device)` 명령으로 모델 객체를 CPU에서 GPU 메모리로 이동. 모든 학습 가능한 파라미터 GPU에 복사.
    * **데이터 배치 이동:** 학습 중 각 배치 입력 데이터와 정답 레이블을 `data.to('cuda')`, `labels.to('cuda')` 명령으로 CPU에서 GPU 메모리로 전송. 데이터 전송 시간 병목 방지 위한 효율적 파이프라인 중요.
4.  **순전파 (Forward Pass) (GPU 연산):**
    * GPU 전송된 입력 데이터 배치, 모델 통과하며 예측값(`output`) 계산.
    * 행렬 곱셈, 합성곱, 활성화 함수 적용 등 모든 복잡 연산 GPU 수많은 코어에서 병렬적 수행, CPU 대비 훨씬 빠름.
5.  **손실 계산 (Loss Calculation) (GPU 연산):**
    * 모델 예측값(`output`)과 실제 정답 레이블(`labels`) 간 오차(Loss) 계산 (예: MSE, CrossEntropyLoss).
    * 손실 계산도 GPU에서 수행.
6.  **역전파 (Backward Pass) (GPU 연산):**
    * 계산된 손실 바탕으로 모델 각 파라미터(가중치)에 대한 기울기(Gradient) 계산.
    * PyTorch `autograd` 기능과 CUDA 활용, 복잡 미분 연산 GPU에서 매우 빠르게 병렬 처리. `loss.backward()` 호출 시 GPU에서 역전파 시작.
7.  **파라미터 업데이트 (Parameter Update) (GPU 연산):**
    * 계산된 기울기(Gradient) 사용, 모델 가중치 업데이트 (예: `optimizer.step()`).
    * 이 과정도 GPU에서 수행, 모델이 데이터 학습하는 핵심 단계.
8.  **반복:**
    * 3단계부터 7단계까지 과정, 정해진 횟수(에폭, Epoch)만큼 반복. 에폭은 전체 데이터셋 한 번 학습 단위.
    * 각 에폭 내에서는 데이터셋을 작은 배치로 나누어 과정 반복.

이러한 GPU 활용 처리 과정은 딥러닝 모델이 대규모 데이터셋을 통해 빠르게 학습하고 높은 성능 달성하도록 돕는 핵심 원동력입니다.

---

## 실습 환경 및 준비물

* **Google 계정:** Google Colab 사용을 위해 필수
* **웹 브라우저:** Chrome 권장 (최적화된 성능)

## 참고사항

* **Colab 시작하기:** Colab은 `*.ipynb` 확장자를 사용하는 Jupyter Notebook 형식의 파일을 사용합니다. 이는 코드 셀과 텍스트(마크다운) 셀을 함께 사용하여 코드와 설명을 동시에 문서화하기에 좋습니다.
* **Colab UI 소개:** Colab 화면 구성은 크게 메뉴바, 도구바, 파일 탐색기, 코드/텍스트 셀 영역으로 나뉩니다. 각 부분의 기능을 익혀두면 효율적인 작업이 가능합니다.
* **Colab에서 GPU 런타임 설정 방법:** 위 실습 내용의 '런타임 유형 변경' 부분을 다시 참고하여 GPU를 활성화했는지 확인하세요. AI 학습에는 GPU가 필수적입니다.
