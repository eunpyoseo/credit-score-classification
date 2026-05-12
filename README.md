# 💳 Credit Score Classification — Deep Learning Competition

신용점수(Good / Standard / Poor) 3등급 분류 딥러닝 프로젝트  
Kaggle Credit Score Classification 데이터를 기반으로 전처리 → EDA → 피처 선택 → TabNet 모델링 전 과정을 수행했습니다.


---


## 📌 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 목표 | 고객 금융 정보로 신용등급(Good / Standard / Poor) 분류 |
| 데이터 | [Kaggle - Credit Score Classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classification) |
| 모델 | Baseline MLP → **TabNet** (최종) |
| 최종 성능 | **Accuracy 80.76% / Weighted F1 0.8076** |
| 환경 | Python 3, PyTorch, pytorch-tabnet, scikit-learn |

---

## 📑 목차 (Table of Contents)
- [1. 🔧 전처리 및 파생변수 생성](#1--전처리-및-파생변수-생성)
- [2. 📊 EDA (탐색적 데이터 분석)](#2--eda)
- [3. 🎯 Feature Selection & 모델 선택](#3--feature-selection--모델-선택)
- [4. 🚀 개선사항 — Baseline MLP 대비 단계별 비교](#4--개선사항--baseline-mlp-대비-단계별-비교)
- [5. 📈 최종 성능 결과](#5--최종-성능-결과)
- [6. ⚙️ 실행 환경](#️-실행-환경)

---

---

## 1. 🔧 전처리 및 파생변수 생성

신용점수는 식별자가 아닌 **상환능력, 상환행동, 신용거래 구조**로 결정된다는 도메인 논리를 기반으로 전처리를 설계했습니다.

### 처리 내용

**① 식별자 제거** — `ID`, `Customer_ID`, `Name`, `SSN`은 예측에 무의미하고 과적합 위험이 있어 제거

**② 대출 유형 분해** (`Type_of_Loan`)  
쉼표로 나열된 텍스트를 → 대출 종류 수(`Num_Loan_Types`) + 8종 대출 보유 여부 플래그(0/1)로 분해  
- 고위험 급전 대출(`Payday Loan`) 보유 시 신용 리스크 상승 반영

**③ 결제 행동 분해** (`Payment_Behaviour`)  
소비 수준(`Spend_Level`: High/Low) + 결제 금액 크기(`Payment_Value`: Small/Medium/Large)로 독립 분리

**④ 도메인 기반 파생변수 생성**

| 파생변수 | 수식 | 의미 |
|----------|------|------|
| `Debt_to_Income` | 미상환부채 ÷ 연소득 | 소득 대비 총 부채 부담 비율 |
| `EMI_to_Salary` | 월 상환액 ÷ 월 급여 | 월 고정 상환 부담 비율 |
| `Delay_Severity` | 연체일수 × 연체 횟수 | 빈도·기간 복합 연체 심각도 |
| `Monthly_Available` | 급여 − 상환액 − 투자액 | 실질 가처분 소득 |
| `Credit_History_Years` | 신용이력 개월 ÷ 12 | 연 단위 신용 거래 기간 |
| `Invest_to_Salary` | 월 투자액 ÷ 월 급여 | 소득 중 투자로 나가는 비율 |
| `Balance_to_Salary` | 월 잔고 ÷ 월 급여 | 소득 대비 유동성 수준 |
| `Avg_Delay_per_Payment` | 연체일수 ÷ (연체 횟수 + 1) | 결제 건당 평균 연체 강도 |
| `Credit_Card_Load` | 신용카드 수 × 이자율 | 카드 기반 이자 부담 종합 지표 |
| `Loan_per_Income` | 대출 건수 ÷ 연소득 | 소득 대비 대출 밀도 |
| `Income_per_Loan` | 연소득 ÷ (대출 건수 + 1) | 대출 건당 감당 가능한 소득 수준 |

**⑤ 데이터 누수 방지** — Imputer·Scaler는 train에만 `fit`, val에는 `transform`만 적용

---

## 2. 📊 EDA
**① 클래스 불균형**  
Standard 53.2% / Poor 29.0% / Good 17.8%로 편향 → Weighted F1, Macro F1, Confusion Matrix를 함께 모니터링

<img width="2244" height="768" alt="image" src="https://github.com/user-attachments/assets/aa2ec144-137f-4fac-a2ad-434464d491de" />

> 📷 `01_class_distribution.png` — 클래스 분포 막대그래프 + 파이차트

---
**② 등급별 수치 차이**  
Poor는 Good 대비 이자율·연체일수·미상환부채가 유의미하게 높고 신용이력은 짧음 → 도메인 파생변수 생성의 타당성을 직접 뒷받침

<img width="2568" height="1568" alt="image" src="https://github.com/user-attachments/assets/b46187bc-2a41-43b2-9ab5-ef3aa0c8021f" />

> 📷 `02_boxplots.png` — 등급별 주요 변수(미상환부채, 이자율, 연체일수, 신용이력) Boxplot

---
**③ Credit_Mix 변별력**  
`Bad` → Poor 비율 압도적, `Good` → Good 비율 높음 → 대출 포트폴리오 건전성이 핵심 분류 변수

<img width="1766" height="768" alt="image" src="https://github.com/user-attachments/assets/9df04119-7847-4f36-9a55-31da6176d32b" />

> 📷 `03_credit_mix_bar.png` — Credit_Mix별 Credit_Score 비율 누적 막대그래프

---
**④ 상관관계 검증**  
`Outstanding_Debt`, `Interest_Rate`, `Delay_Severity` → 신용점수와 **음의 상관**  
`Credit_History_Age`, `Monthly_Available` → **양의 상관** (EDA 논리와 일관)

<img width="1877" height="1368" alt="image" src="https://github.com/user-attachments/assets/de3d93c2-da77-4726-aa91-c7feee6c84f0" />

> 📷 `04_correlation_heatmap.png` — 주요 변수 상관관계 히트맵

---

## 3. 🎯 Feature Selection & 모델 선택

### Feature Selection

RandomForest (`n_estimators=120`, `max_depth=14`, `class_weight='balanced_subsample'`)의 `feature_importances_`를 기준으로 **중요도 0.001 미만 변수를 노이즈로 제거**, 최종 **51개** 피처 선택

<img width="1969" height="1568" alt="image" src="https://github.com/user-attachments/assets/a07a64c3-2328-462f-9e81-d898edf0b4bd" />
> 📷 `05_rf_feature_importance.png` — RandomForest 기반 Feature Importance Top 25

### 모델 선택: TabNet

| 비교 항목 | MLP | **TabNet** |
|-----------|-----|-----------|
| 피처 상호작용 | 암묵적 독립 가정 | Sequential Attention으로 비선형 상호작용 포착 |
| 설명력 | 낮음 | Feature Importance 자동 추출 가능 |
| 클래스 불균형 | 별도 처리 필요 | `weights=1`로 자동 class balancing |
| 과적합 방지 | 수동 설정 | `patience=20` early stopping |

### 하이퍼파라미터

```python
TabNetClassifier(
    n_d=32, n_a=32,           # 표현 차원
    n_steps=4,                 # attention step 수 (균형점)
    gamma=1.5,
    mask_type='entmax',
    optimizer_params=dict(lr=1e-2, weight_decay=1e-5),
    scheduler_fn=StepLR,       # 후반 lr 감소로 수렴 안정화
    scheduler_params={'step_size': 20, 'gamma': 0.9},
    max_epochs=150, patience=20, batch_size=2048
)
```

<img width="1969" height="1368" alt="image" src="https://github.com/user-attachments/assets/5ff30c7c-8821-49d3-b6c2-5e43043c221e" />
> 📷 `06_tabnet_feature_importance.png` — TabNet Attention 기반 Feature Importance Top 25

---

## 4. 🚀 개선사항 — Baseline MLP 대비 단계별 비교

> **Baseline** : 원본 피처 + LabelEncoding + 단순 MLP (64→32→3, 20 epoch, 클래스 가중치 없음)

| 개선 단계 | 내용 |
|-----------|------|
| ① 데이터 누수 제거 | val을 scaler fit에서 완전 분리 → 과낙관 성능 방지 |
| ② 파생변수 추가 | Debt_to_Income, EMI_to_Salary 등 비율 지표 22개 → 재무 위험 직접 표현 |
| ③ 인코딩 개선 | Type_of_Loan 분해 + One-Hot Encoding → LabelEncoding의 잘못된 순서 왜곡 제거 |
| ④ 클래스 불균형 보정 | `class_weight='balanced'`, `weights=1` → 소수 클래스(Good) 적극 학습 |
| ⑤ 모델 전략 | TabNet + 조기종료 + StepLR → 과적합 없이 성능 향상 |

---

## 5. 📈 최종 성능 결과

### 모델 비교

| 모델 | Accuracy | Weighted F1 | Macro F1 |
|------|----------|-------------|----------|
| Baseline MLP | 71.58% | 0.7155 | - |
| **Final TabNet** | **80.76%** | **0.8076** | **0.8060** |
| **개선폭** | **+9.18%p** | **+0.0921** | - |

### Validation Score 세부 결과

```
Accuracy    : 80.76%
Weighted F1 : 0.8076
Macro F1    : 0.8060
```

- 소수 클래스 **Good Recall 0.86** → 실제 우량 고객을 놓치는 오류 최소화
- 다수 클래스 **Standard Precision 0.88** → 과예측 억제
- 세 등급 모두 F1 0.80 이상 → 클래스 불균형 속에서도 균등한 예측력 확보
- 권장 기준(75%) **상회**

<img width="1333" height="969" alt="image" src="https://github.com/user-attachments/assets/fc7bd105-8586-4284-8376-fbb25dc2b21d" />
> 📷 `07_confusion_matrix.png` — Confusion Matrix (Acc 80.76% | Weighted F1 0.8076)

---

## ⚙️ 실행 환경

```
Python       3.x
PyTorch      (Apple Silicon MPS / CUDA / CPU 자동 감지)
pytorch-tabnet
scikit-learn
pandas, numpy, seaborn, matplotlib
```

```bash
pip install pytorch-tabnet scikit-learn pandas numpy seaborn matplotlib
```
