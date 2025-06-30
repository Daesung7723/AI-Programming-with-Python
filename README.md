# 파이토치 및 LLM 기반 AI 서비스 개발 과정

### Ⅰ. 과정 개요

본 과정은 파이토치(PyTorch)와 구글 코랩(Google Colab)을 사용한다. 2주 동안 딥러닝 이론, LLM 파인튜닝, 상용 API 활용법을 학습하고 AI 서비스 개발 전체 과정을 다룬다. 수강생의 상용 AI 소프트웨어 개발 능력 확보를 목표로 한다.

### Ⅱ. 학습 목표

- 파이토치, 구글 코랩 기반 개발 환경 구축
- DNN, CNN, RNN 모델 구조 이해 및 구현
- Hugging Face 라이브러리 기반 언어 모델 파인튜닝
- 임베딩 및 벡터 데이터베이스 개념 이해와 활용
- 상용 LLM API 활용 AI 서비스 개발
- 팀 프로젝트를 통한 AI 서비스 개발 과정 전체 수행

---

### Ⅲ. 상세 교육 과정 (10일 / 80시간)

### **1주차: 딥러닝 기초 및 핵심 모델**

---

#### **1일차: 과정 소개 및 개발 환경**

- AI, 머신러닝, 딥러닝 개념 및 역사
- 구글 Colab 사용법 (GPU, 파일 시스템, 명령어)
- Numpy, Pandas, Matplotlib 사용법 실습

#### **2일차: 파이토치 기본**

- 텐서(Tensor) 자료구조 이해 및 연산
- 자동 미분(Autograd)과 경사하강법(Gradient Descent) 원리
- 파이토치 기반 선형 회귀(Linear Regression) 모델 구현

#### **3일차: DNN (심층 신경망)**

- 다층 퍼셉트론(MLP) 구조, 활성화 함수, 손실 함수, 옵티마이저
- 과적합(Overfitting) 원인 및 방지 기법 (Regularization, Dropout)
- DNN 분류 모델 구축 (데이터셋: 패션 MNIST)

#### **4일차: CNN (합성곱 신경망)**

- 합성곱(Convolution), 풀링(Pooling) 연산 원리
- 주요 CNN 아키텍처 구조 (LeNet-5, VGGNet, ResNet)
- CNN 분류 모델 구축 (데이터셋: CIFAR-10)

#### **5일차: RNN (순환 신경망) 및 NLP 기초**

- RNN, LSTM, GRU 구조 및 순차 데이터 처리
- 텍스트 데이터 표현법 (단어 임베딩 - Word2Vec)
- LSTM 기반 텍스트 분류 모델 구현 (데이터셋: IMDB 영화 리뷰)

<br>

### **2주차: 모델 심화 및 서비스 개발**

---

#### **6일차: 문장 임베딩 및 의미 검색**

- 문맥 기반 임베딩 (Sentence-BERT) 원리
- 벡터 데이터베이스(Vector DB) 개념 및 용도
- SBERT 기반 문장 유사도 측정 및 검색 기능 구현

#### **7일차: 전이 학습 및 파인튜닝**

- 이미지 모델 전이 학습(Transfer Learning) 방법
- Hugging Face 라이브러리(Transformers, Datasets) 사용법
- PEFT(LoRA) 기법 기반 한국어 LLM 파인튜닝 실습

#### **8일차: 상용 LLM API 활용**

- API 기반 AI 서비스 개발 아키텍처
- Prompt Engineering 기법
- 상용 LLM (OpenAI, Google API) 연동 프로그램 개발 (Q&A, 요약)
- 생성 모델(GAN), MLOps 개념 소개

#### **9일차: 최종 프로젝트 - 개발**

- 주제 선정 및 시스템 설계
- 데이터 수집 및 처리, 모델 개발 및 기능 구현
- 팀별 기술 멘토링 및 문제 해결

#### **10일차: 최종 프로젝트 - 발표**

- 모델 성능 개선 및 기능 보완
- 팀별 개발 결과물 발표
- 과정 정리 및 질의응답


---
---

# PyTorch and LLM-based AI Service Development Course

### I. Course Overview

This is a 2-week (10 days, 80 hours) intensive AI service development training program that utilizes PyTorch and Google Colab. The course covers the entire process of AI service development, from core deep learning theories to fine-tuning the latest language models (LLMs) and developing services using commercial APIs. The goal is for students to acquire the skills necessary to develop commercial AI software.

### II. Learning Objectives

* Confidently build and use a deep learning development environment with PyTorch and Google Colab.

* Understand the architecture of core deep learning models like DNN, CNN, and RNN, and be able to implement them directly.

* Fine-tune the latest language models for specific purposes using the Hugging Face library.

* Understand and utilize the concepts of embeddings and vector databases for semantic search systems.

* Develop AI services with various features such as Q&A and summarization using commercial LLM APIs.

* Experience the full cycle of AI service development, from ideation to final presentation, through a team project.

### III. Detailed Curriculum (10 Days / 80 Hours)

### **Week 1: Deep Learning Fundamentals and Core Models**

#### **Day 1: Course Introduction & Development Environment Setup**

* **Key Topics:** Concepts and historical development of AI, Machine Learning, and Deep Learning.

* **Hands-on Practice:**

  * Advanced Google Colab usage (GPU settings, file system integration, essential commands).

  * Core functionalities of Numpy, Pandas, and Matplotlib libraries.

#### **Day 2: PyTorch Fundamentals**

* **Key Topics:**

  * Understanding and operating on Tensors, the core data structure of PyTorch.

  * Principles of the Autograd system and Gradient Descent.

* **Hands-on Practice:** Implementing a Linear Regression model from scratch using PyTorch.

#### **Day 3: Mastering DNN (Deep Neural Network)**

* **Key Topics:**

  * Structure of Multi-Layer Perceptrons (MLP), types and roles of activation functions, loss functions, and optimizers.

  * Causes of and solutions for Overfitting (Regularization, Dropout).

* **Hands-on Practice:** Building a DNN image classification model using the Fashion MNIST dataset.

#### **Day 4: Conquering CNN (Convolutional Neural Network)**

* **Key Topics:**

  * Principles of Convolution and Pooling operations and their role in visual data processing.

  * Comparison of structural features of major CNN architectures (LeNet-5, VGGNet, ResNet).

* **Hands-on Practice:** Building a CNN image classification model using the CIFAR-10 dataset.

#### **Day 5: RNN (Recurrent Neural Network) & NLP Basics**

* **Key Topics:**

  * Structures of RNN, LSTM, and GRU and their methods for processing sequential data.

  * Text data representation for Natural Language Processing (Word Embedding - Word2Vec).

* **Hands-on Practice:** Implementing an LSTM-based text sentiment classification model using the IMDB movie review dataset.

### **Week 2: Advanced Models and Service Development**

#### **Day 6: Sentence Embedding and Semantic Search**

* **Key Topics:**

  * Principles of context-aware embedding models (Sentence-BERT).

  * Vector Databases for efficient storage and retrieval of large-scale vector data.

* **Hands-on Practice:** Implementing sentence similarity measurement and a semantic search feature using Sentence-BERT.

#### **Day 7: Transfer Learning & LLM Fine-Tuning**

* **Key Topics:**

  * Transfer Learning techniques using pre-trained image models.

  * Essential usage of the Hugging Face ecosystem (Transformers, Datasets).

  * Efficient model tuning with PEFT (Parameter-Efficient Fine-Tuning) and LoRA techniques.

* **Hands-on Practice:** Fine-tuning a pre-trained Korean LLM for a specific domain.

#### **Day 8: Developing Services with Commercial LLM APIs**

* **Key Topics:**

  * Architectural design for API-based AI services.

  * Prompt Engineering techniques to maximize LLM performance.

  * Introduction to Generative Models (GANs) and MLOps concepts.

* **Hands-on Practice:** Developing Q&A and text summarization programs by integrating commercial LLM APIs (e.g., OpenAI, Google).

#### **Day 9: Final Project - Development**

* **Key Topics:** Team-based AI service development project.

* **Activities:**

  * Project theme selection and system architecture design.

  * Data collection, preprocessing, model development, and core feature implementation.

  * Problem-solving and development guidance through team-specific technical mentoring.

#### **Day 10: Final Project - Presentation & Wrap-up**

* **Key Topics:** Project showcase and course conclusion.

* **Activities:**

  * Improving the performance of the developed model and enhancing service features.

  * Final presentation and demo of each team's project.

  * Course review and Q&A session.
