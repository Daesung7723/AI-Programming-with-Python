# [1주차 1일] AI 개요 및 파이썬 개발 환경 구축

## 1. 과정 소개 및 목표 설정

안녕하세요! "파이토치 및 LLM 기반 AI 서비스 개발 과정"에 오신 것을 환영합니다. 
이번 1일차 강의에서는 앞으로의 긴 여정을 위한 첫걸음을 내딛습니다. 인공지능의 큰 그림을 이해하고, 우리의 강력한 무기가 될 개발 도구들과 친해지는 시간을 갖겠습니다.

- **본 과정의 최종 목표:** 수강생 스스로 상용 수준의 AI 소프트웨어를 개발하는 능력을 확보하는 것입니다.
- **주 개발 환경:** 본 과정에서는 파이토치(PyTorch)를 메인 프레임워크로, 구글 코랩(Google Colab)을 실습 플랫폼으로 사용합니다.

## 2. 인공지능(AI), 머신러닝, 딥러닝의 이해

### 2-1. 이론: 주요 개념 및 관계

인공지능, 머신러닝, 딥러닝은 종종 혼용되지만, 명확한 포함 관계를 가집니다.

- **인공지능 (Artificial Intelligence, AI):** 가장 넓은 범위의 개념입니다. 인간의 지능적인 행동을 모방하는 모든 컴퓨터 과학 기술을 의미합니다.
- **머신러닝 (Machine Learning, ML):** 인공지능의 하위 분야입니다. 컴퓨터가 명시적인 프로그램 없이 데이터를 통해 스스로 학습하여 패턴을 찾고 예측을 수행하는 기술입니다.
- **딥러닝 (Deep Learning, DL):** 머신러닝의 한 분야로, '인공신경망(Artificial Neural Network)'을 깊게(Deep) 쌓아올려 구현합니다.

**[관계도]**
> `인공지능 > 머신러닝 > 딥러닝`

![AI-ML-DL](https://i.imgur.com/S2Y2pNd.png)

## 3. 개발 환경: 구글 코랩(Google Colab) 활용법

구글 코랩은 클라우드 기반의 무료 Jupyter Notebook 환경으로, 별도의 설치 없이 GPU를 사용할 수 있는 큰 장점이 있습니다.

### 실습: 코랩 환경 마스터하기

#### 1) GPU 설정 확인
- `런타임` -> `런타임 유형 변경` -> `GPU` 선택 후, 아래 `[CODE-1]`을 실행하여 GPU 할당을 확인합니다.

`[CODE-1]`
```python
!nvidia-smi
```


#### 2) Google Drive 마운트
- 내 구글 드라이브를 코랩 환경에 연결하려면 아래 `[CODE-2]`를 실행합니다.

`[CODE-2]`
```python
from google.colab import drive
drive.mount('/content/drive')
```

## 4. 데이터 분석을 위한 파이썬 핵심 라이브러리 실습

### 4-1. Numpy (넘파이): 수치 데이터의 기반
- 파이썬에서 대규모 다차원 배열과 행렬 연산을 빠르고 효율적으로 처리하는 라이브러리입니다.
- 아래 `[CODE-3]`으로 실습합니다.

`[CODE-3]`
```python
import numpy as np

arr1 = np.array([1, 2, 3, 4, 5])
print(f"1차원 배열: {arr1}")

arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print(f"2차원 배열:\n{arr2}")

print(f"arr2 형태: {arr2.shape}")
```

### 4-2. Pandas (판다스): 데이터 가공 및 분석의 핵심
- 행과 열로 이루어진 테이블 형태의 데이터를 다루는 데 특화된 라이브러리입니다.
- 아래 `[CODE-4]`로 실습합니다.

`[CODE-4]`
```python
import pandas as pd

data = {
    '이름': ['김교수', '이학생', '박조교', '최박사'],
    '나이': [45, 24, 31, 52],
    '전공': ['AI', '경영', '컴퓨터공학', 'AI']
}
df = pd.DataFrame(data)
print("생성된 DataFrame:")
print(df)
```

### 4-3. Matplotlib (맷플롯립): 데이터 시각화의 표준
- 데이터를 그래프나 차트로 시각화하여 직관적으로 이해할 수 있게 돕는 라이브러리입니다.

#### 한글 폰트 설치
- Colab 환경에서 Matplotlib 그래프에 한글을 표시하기 위해 폰트를 설치합니다.
- **아래 `[CODE-5]`를 실행한 후에는 반드시 [런타임] -> [런타임 다시 시작]을 눌러주세요.**

`[CODE-5]`
```python
!sudo apt-get install -y fonts-nanum
!sudo fc-cache -fv
!rm ~/.cache/matplotlib -rf
```


#### 그래프 그리기
- 런타임 다시 시작 후, 아래 `[CODE-6]`을 실행하여 그래프를 확인합니다.

`[CODE-6]`
```python
import matplotlib.pyplot as plt
import numpy as np

# 한글 폰트 설정
plt.rc('font', family='NanumBarunGothic') 

# 데이터 준비
x = np.arange(1, 11)
y = x * 2

# 선 그래프 그리기
plt.figure(figsize=(10, 4))
plt.plot(x, y, 'go--', label='가격')
plt.title('월별 가격 변동')
plt.xlabel('월(Month)')
plt.ylabel('가격(Price)')
plt.legend()
plt.grid(True)
plt.show()
```


## 5. 학습 성취도 확인 및 응용

### 5-1. 퀴즈

1. AI, 머신러닝, 딥러닝의 관계를 바르게 설명한 것은?
    - (c) AI > 머신러닝 > 딥러닝
2. 데이터로부터 컴퓨터가 스스로 학습하여 예측을 수행하는 기술을 무엇이라고 하는가?
    - (머신러닝)
3. 수치 연산과 다차원 배열을 효율적으로 처리하기 위한 파이썬 라이브러리는?
    - (Numpy)
4. Pandas에서 테이블 형태의 2차원 데이터를 다루는 핵심 자료구조의 이름은?
    - (DataFrame)
5. 데이터 분석 결과를 그래프로 시각화하여 직관적으로 이해하도록 돕는 라이브러리는?
    - (Matplotlib)

### 5-2. 미니 프로젝트 ①: "공공데이터를 활용한 서울시 인구 현황 분석"

- **개요:** 오늘 배운 Pandas로 데이터를 읽고 가공하며, Matplotlib으로 분석 결과를 시각화하는 간단한 프로젝트입니다.
- **실습 안내:** 아래 `[CODE-7]`을 복사하여 Colab 셀에 붙여넣고 실행해봅니다.

`[CODE-7]`
```python
import pandas as pd
import matplotlib.pyplot as plt

# STEP 1: 이 셀을 실행하여 샘플 데이터(seoul_population.csv)를 생성하세요.
%%writefile seoul_population.csv
자치구,인구수,면적
강남구,547453,39.5
서초구,423340,46.9
송파구,673926,33.8
강동구,469496,24.5
마포구,382494,23.8
영등포구,404595,24.5
강서구,585901,41.4

# STEP 2: 데이터를 읽고 분석, 시각화하는 코드입니다.
# 런타임을 다시 시작했다면 폰트 설정을 다시 해야 합니다.
plt.rc('font', family='NanumBarunGothic') 

df_seoul = pd.read_csv('seoul_population.csv')
df_sorted = df_seoul.sort_values(by='인구수', ascending=False)

plt.figure(figsize=(12, 6))
plt.bar(df_sorted['자치구'], df_sorted['인구수'], color='dodgerblue')
plt.title('서울시 주요 자치구별 인구수', fontsize=16)
plt.xlabel('자치구', fontsize=12)
plt.ylabel('인구수 (명)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
```


## 6. 정리 및 다음 차시 예고

- **1일차 학습 내용 요약:**
    - AI, 머신러닝, 딥러닝의 개념과 관계를 이해했습니다.
    - 구글 코랩을 통해 GPU 개발 환경을 설정하고 사용하는 법을 익혔습니다.
    - 데이터 분석의 필수 3종 라이브러리인 Numpy, Pandas, Matplotlib의 기초 사용법을 실습했습니다.

- **2일차 학습 예고:**
    - 드디어 딥러닝 프레임워크인 **파이토치(PyTorch)**의 세계로 들어갑니다.
    - 딥러닝의 기본 데이터 단위인 **텐서(Tensor)**의 개념을 이해하고, 모델 학습의 핵심 원리인 **경사하강법(Gradient Descent)**에 대해 학습할 것입니다.

오늘 하루 고생 많으셨습니다!