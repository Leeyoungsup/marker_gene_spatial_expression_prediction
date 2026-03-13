# Marker Gene Spatial Expression Prediction

H&E 염색 병리 이미지로부터 marker gene의 spatial expression을 예측하는 딥러닝 모델 (3-Head: Presence + Expression + Uncertainty)

---

## 1. Introduction

### Background
- Spatial transcriptomics 데이터는 비용과 실험 제약으로 대규모 확보가 어려움
- 조직의 morphology는 cell type composition, tumor microenvironment, immune infiltration 등과 강한 상관관계를 가짐
- H&E 이미지의 형태학적 특징만으로 유전자 발현의 공간적 분포를 추정할 수 있는 가능성 존재
- 기존 regression 방식은 **zero-inflated 특성**(대부분의 gene이 특정 패치에서 미발현)을 고려하지 못함

### Objective
- **H&E patch → gene expression vector** 형태의 regression 모델 학습
- Spot-level에서 19개 marker gene의 **presence (존재 확률)**, **expression (조건부 발현량)**, **uncertainty (예측 불확실성)** 동시 예측
- **Zero-Inflated Heteroscedastic Regression** 프레임워크 적용
- Xenium spatial transcriptomics 데이터 기반 학습 및 평가

### Target Marker Genes (19 genes)

| Category | Genes |
|---|---|
| Epithelial | EPCAM, EGFR |
| Fibroblast / Stromal | ACTA2, PDGFRA, PDGFRB, SFRP4 |
| Endothelial | PECAM1 |
| Macrophage / Myeloid | CD68, AIF1, FCGR3A, MRC1 |
| T Cell | CD3E, CD4, CD8A, TRAC |
| B Cell | CD79A, MS4A1, BANK1, TCL1A |

| Gene | Description |
|---|---|
| EPCAM | Epithelial Cell Adhesion Molecule. 상피세포 표면에 발현되는 adhesion molecule로 epithelial cell 간 결합 유지에 관여하며 carcinoma에서 높은 발현을 보이는 대표적인 epithelial marker. |
| EGFR | Epidermal Growth Factor Receptor. 세포 증식과 분화를 조절하는 receptor tyrosine kinase로 다양한 epithelial tumor에서 overexpression 또는 mutation이 나타나는 성장 신호 전달 수용체. |
| ACTA2 | Alpha Smooth Muscle Actin. myofibroblast와 smooth muscle cell에서 발현되는 cytoskeletal protein으로 조직 수축 및 ECM remodeling에 관여하며 CAF (cancer-associated fibroblast) marker로 사용됨. |
| PDGFRA | Platelet-Derived Growth Factor Receptor Alpha. fibroblast proliferation과 migration을 조절하는 receptor tyrosine kinase로 stromal cell 및 mesenchymal lineage marker로 사용됨. |
| PDGFRB | Platelet-Derived Growth Factor Receptor Beta. pericyte 및 fibroblast에서 발현되는 receptor로 혈관 안정성 유지와 stromal remodeling 및 angiogenesis에 관여. |
| SFRP4 | Secreted Frizzled-Related Protein 4. Wnt signaling pathway를 조절하는 단백질로 stromal fibroblast 및 CAF에서 발현되며 조직 remodeling과 tumor progression과 관련됨. |
| PECAM1 | Platelet Endothelial Cell Adhesion Molecule 1 (CD31). 혈관 내피세포 adhesion molecule로 angiogenesis와 leukocyte transmigration에 관여하며 microvessel density 평가에 사용되는 대표적인 endothelial marker. |
| CD68 | macrophage lysosomal glycoprotein으로 monocyte 및 macrophage에서 발현되며 tumor microenvironment에서 tumor-associated macrophage (TAM) marker로 사용됨. |
| AIF1 | Allograft Inflammatory Factor 1 (IBA1). macrophage와 microglia activation marker로 inflammatory response 및 immune activation 과정에 관여. |
| FCGR3A | Fc Gamma Receptor IIIA (CD16). antibody-dependent cellular cytotoxicity (ADCC)에 관여하는 receptor로 macrophage, NK cell 등에서 발현됨. |
| MRC1 | Mannose Receptor C-Type 1 (CD206). M2 macrophage marker로 pathogen recognition과 endocytosis에 관여하며 anti-inflammatory immune response와 조직 재형성 과정에 관련됨. |
| CD3E | T-cell receptor complex 구성 단백질로 모든 mature T cell에서 발현되며 T cell lineage identification에 사용되는 대표적인 marker. |
| CD4 | helper T cell surface glycoprotein으로 MHC class II와 결합하여 antigen recognition 및 immune response 조절에 관여. |
| CD8A | cytotoxic T lymphocyte marker로 MHC class I antigen recognition에 관여하며 tumor cell killing과 antiviral immune response에서 중요한 역할 수행. |
| TRAC | T Cell Receptor Alpha Constant region을 코딩하는 유전자로 T cell receptor 구조 형성에 필수적이며 T cell lineage marker로 사용됨. |
| CD79A | B cell receptor complex 구성 단백질로 B cell activation 및 signaling에 필수적이며 B cell lineage marker로 널리 사용됨. |
| MS4A1 | CD20 단백질을 코딩하는 유전자로 mature B cell surface marker이며 B-cell lymphoma 치료제 rituximab의 주요 target. |
| BANK1 | B Cell Scaffold Protein with Ankyrin Repeats 1. B cell receptor signaling pathway를 조절하는 scaffold protein으로 B cell activation 조절에 관여. |
| TCL1A | T Cell Leukemia/Lymphoma 1A. 초기 B cell 및 germinal center B cell에서 발현되는 단백질로 lymphocyte proliferation과 발달 과정에 관여. |

---

## 2. Method

### 2.1 Data

- **데이터 소스**: Xenium spatial transcriptomics
- **WSI 분할**: 20x 기준 512×512 non-overlapping grid
- **패치 포함 조건**: 패치 내 ST spot ≥ 1개
- **Train/Val split**: 80/20 (random_state=42)

<!-- TODO: 데이터 구성 이미지 삽입 -->

### 2.2 Model Input (Multi-Scale Patch)

| Scale | Resolution | Shape | Description |
|---|---|---|---|
| High-resolution | 20x | (3, 512, 512) | 세포 morphology 정보 |
| Context | 5x | (3, 512, 512) | 넓은 조직 구조 정보 (동일 중심, ~4배 넓은 영역) |

### 2.3 Model Architecture (3-Head)

- **Encoder**: ConvNeXt-Tiny (`tu-convnext_tiny`, ImageNet pretrained) × 2 (20x, 5x 각각)
- **Feature Fusion**: 두 encoder의 last-stage feature를 element-wise addition
- **Pooling**: Global Average Pooling → (B, 768)
- **Shared FC**: Linear(768→512) → ReLU → Dropout(0.2)
- **Three-Head 구조**:
  - **Head A (Presence)**: Linear(512→256) → ReLU → Linear(256→19) — logits (BCE loss 적용)
  - **Head B (Expression)**: Linear(512→256) → ReLU → Linear(256→19) → Sigmoid — 조건부 발현량 [0, 1]
  - **Head C (Uncertainty)**: Linear(512→256) → ReLU → Linear(256→19) — unconstrained (softplus로 σ 변환)

```
                    ┌─ Head A: Presence logits ─── sigmoid → P(gene expressed)
                    │
20x ─→ Encoder ─┐  │
                 ├─ Element-wise Add ─→ GAP ─→ Shared FC(768→512) ─┼─ Head B: Expression μ ──── sigmoid → E[expr | expressed]
5x ──→ Encoder ─┘                                                  │
                    └─ Head C: Log Variance ────── softplus → σ (uncertainty)
```

**Final Prediction**:
```
y_pred = sigmoid(presence_logits) × expression_mu
uncertainty = softplus(log_var)
```

<!-- TODO: 모델 아키텍처 다이어그램 이미지 삽입 -->

### 2.4 Label Generation

- **Head A — Presence** (존재 확률, binary {0, 1})
  ```
  presence[g] = 1.0  if gene_count[g] > 0  else  0.0
  ```
  - 해당 패치 내 gene g의 transcript가 1개 이상 존재하면 1, 아니면 0

- **Head B — Expression** (조건부 발현량, range [0, 1])
  ```
  expression[g] = clip(log1p(transcript_count), 0, 10) / 10
  ```
  - Xenium은 imaging 기반 기술로 CPM 정규화 불필요
  - 학습 시 presence=1인 gene에 대해서만 loss 계산 (masked)

- **Head C — Uncertainty** (label 불필요)
  - 별도의 GT label 없이, loss를 통해 자동 학습
  - 모델이 σ = softplus(log_var)를 출력하며, |prediction error|와의 calibration으로 학습

### 2.5 Loss Function (Decoupled 4-Term Loss)

```
Total Loss = α × BCE(presence_logits, gt_presence)
           + β × masked_SmoothL1(expression_mu, gt_expression)
           + δ × (1 − mean_PCC(expression_mu, gt_expression))
           + γ × SmoothL1(σ, |expression_error|.detach())
```

| Component | Function | Weight | Description |
|---|---|---|---|
| Presence Loss | BCEWithLogitsLoss | α = 1.0 | gene 존재 여부 binary classification |
| Expression Loss | masked SmoothL1Loss | β = 1.0 | presence=1인 gene만 발현량 regression |
| PCC Loss | 1 − mean(Pearson Correlation) | δ = 0.5 | 유전자 간 상관 패턴 최적화 |
| Uncertainty Loss | SmoothL1Loss(σ, \|error\|) | γ = 0.5 | σ가 실제 오차를 추정하도록 calibration |

- 모든 loss term ≥ 0 (학습 안정성 보장)
- σ = softplus(raw_sigma), |error|은 `.detach()`로 gradient 차단

### 2.6 Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 2e-4 |
| Batch Size | 16 |
| Max Epochs | 1000 |
| Gradient Clipping | max_norm = 1.0 |

### 2.7 Data Augmentation

- **Geometric** (20x, 5x 동일 적용): Horizontal flip, Vertical flip, 90° rotation
- **Color** (20x, 5x 독립 적용): Brightness, Contrast, Hue shift, Saturation, Gamma correction, Channel-wise brightness, Gaussian noise, Gaussian blur

### 2.8 Evaluation Metrics

#### PCC (Pearson Correlation Coefficient)

- 두 변수 간 **선형 상관관계의 강도와 방향**을 측정하는 지표
- 범위: **[-1, 1]**
  - 1에 가까울수록 강한 양의 상관 (예측이 GT와 일치하는 방향)
  - 0에 가까울수록 상관관계 없음
  - -1에 가까울수록 강한 음의 상관
- 수식:

$$
\text{PCC} = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2} \cdot \sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}
$$

  - $x_i$: 예측값, $y_i$: GT 값, $\bar{x}$, $\bar{y}$: 각각의 평균

- 본 프로젝트에서 3가지 수준으로 측정:
  - **Per-Gene PCC**: 각 유전자별로 전체 샘플에 대한 PCC 계산 → 유전자별 예측 성능 비교
  - **Per-Sample PCC**: 각 샘플별로 19개 유전자 벡터에 대한 PCC 계산 → 개별 패치의 예측 품질 평가
  - **Global PCC**: 전체 예측값과 GT를 flatten하여 단일 PCC 계산 → 모델 전반적 성능 요약
- MAE와 달리 **스케일에 무관**하게 예측 패턴의 일치도를 평가할 수 있음
- Loss function에도 PCC 기반 loss (1 − mean PCC)를 포함하여 직접 상관관계를 최적화

#### MAE (Mean Absolute Error)

- 예측값과 GT 간 **절대 오차의 평균**
- 범위: **[0, ∞)** (낮을수록 좋음)
- Per-gene 수준으로 계산하여 유전자별 예측 정확도 비교

#### Presence Accuracy

- Presence head의 binary classification 정확도 (threshold = 0.5)
- 각 gene별 정확도 및 전체 평균으로 측정

#### Uncertainty Calibration

- 학습된 σ가 실제 prediction error (|pred − GT|)와 얼마나 일치하는지 평가
- σ vs |error| scatter plot 및 binned calibration curve로 시각화

---

## 3. Result

### 3.1 Per-Gene Performance

| Gene | Pres_Acc | PCC_Pres | PCC_Expr | MAE_Expr | PCC_Comb | MAE_Comb | Mean_σ |
|---|---|---|---|---|---|---|---|
| EPCAM | 0.9335 | 0.7661 | 0.9564 | 0.0512 | 0.9552 | 0.0487 | 0.0592 |
| EGFR | 0.9338 | 0.8076 | 0.9423 | 0.0412 | 0.9413 | 0.0399 | 0.0488 |
| ACTA2 | 0.9453 | 0.7106 | 0.9563 | 0.0497 | 0.9565 | 0.0494 | 0.0537 |
| PDGFRA | 0.9441 | 0.7602 | 0.9428 | 0.0441 | 0.9421 | 0.0437 | 0.0476 |
| PDGFRB | 0.9429 | 0.7637 | 0.9566 | 0.0408 | 0.9562 | 0.0401 | 0.0451 |
| SFRP4 | 0.9116 | 0.7384 | 0.9129 | 0.0674 | 0.9121 | 0.0666 | 0.0734 |
| PECAM1 | 0.9409 | 0.6794 | 0.9493 | 0.0470 | 0.9480 | 0.0468 | 0.0487 |
| CD68 | 0.9368 | 0.7640 | 0.9572 | 0.0431 | 0.9566 | 0.0420 | 0.0485 |
| AIF1 | 0.9358 | 0.7071 | 0.9579 | 0.0410 | 0.9569 | 0.0402 | 0.0416 |
| FCGR3A | 0.9280 | 0.7741 | 0.9477 | 0.0457 | 0.9474 | 0.0437 | 0.0471 |
| MRC1 | 0.9350 | 0.7654 | 0.9123 | 0.0512 | 0.9119 | 0.0505 | 0.0555 |
| CD3E | 0.9091 | 0.7281 | 0.9404 | 0.0511 | 0.9409 | 0.0501 | 0.0548 |
| CD4 | 0.9427 | 0.7340 | 0.9590 | 0.0400 | 0.9585 | 0.0390 | 0.0401 |
| CD8A | 0.9118 | 0.7800 | 0.9296 | 0.0500 | 0.9292 | 0.0485 | 0.0556 |
| TRAC | 0.9015 | 0.7032 | 0.9430 | 0.0553 | 0.9438 | 0.0536 | 0.0571 |
| CD79A | 0.8153 | 0.6827 | 0.9350 | 0.0525 | 0.9345 | 0.0492 | 0.0668 |
| MS4A1 | 0.8440 | 0.7090 | 0.9429 | 0.0528 | 0.9425 | 0.0503 | 0.0601 |
| BANK1 | 0.9058 | 0.7446 | 0.9386 | 0.0488 | 0.9387 | 0.0466 | 0.0535 |
| TCL1A | 0.7787 | 0.6250 | 0.9249 | 0.0473 | 0.9237 | 0.0449 | 0.0494 |
| **Mean** | **0.9103** | **0.7338** | **0.9424** | **0.0484** | **0.9419** | **0.0470** | **0.0530** |

### 3.2 Summary

- **Presence Head**: 평균 Accuracy **0.9103**, 평균 PCC **0.7338**
- **Expression Head**: 평균 PCC **0.9424**, 평균 MAE **0.0484**
- **Combined (P × μ)**: 평균 PCC **0.9419**, 평균 MAE **0.0470**
- **Uncertainty**: 평균 σ **0.0530**
- **Global PCC (flatten)**: Presence **0.7507**, Expression **0.9529**, Combined **0.9527**

### 3.3 Gene Group Performance

| Gene Group | PCC_Pres | PCC_Comb | Mean_σ | Pres_Acc |
|---|---|---|---|---|
| Epithelial (EPCAM, EGFR) | 0.7868 | 0.9482 | 0.0540 | 0.9336 |
| Stromal (ACTA2, PDGFRA, PDGFRB, SFRP4) | 0.7432 | 0.9417 | 0.0549 | 0.9360 |
| Endothelial (PECAM1) | 0.6794 | 0.9480 | 0.0487 | 0.9409 |
| Macrophage (CD68, AIF1, FCGR3A, MRC1) | 0.7527 | 0.9432 | 0.0482 | 0.9339 |
| T Cell (CD3E, CD4, CD8A, TRAC) | 0.7363 | 0.9431 | 0.0519 | 0.9163 |
| B Cell (CD79A, MS4A1, BANK1, TCL1A) | 0.6903 | 0.9349 | 0.0574 | 0.8359 |

### 3.4 Figures

#### Per-Gene PCC & Uncertainty Bar Chart
![유전자별 PCC & Uncertainty](fig/유전자별%20PCC%20&%20MAE%20막대%20차트.png)

#### Scatter Plot: Combined Pred vs GT (with uncertainty coloring)
![Scatter Plot Combined](fig/Scatter%20Plot:%20Pred%20vs%20GT(Combined%20=%20P%20×%20μ,%20uncertainty%20color).png)

#### Error Distribution Boxplot (Combined)
![오차 분포 Boxplot](fig/오차%20분포%20Boxplot.png)

#### Gene Group Performance Summary
![유전자 그룹별 성능 요약](fig/유전자%20그룹별%20성능%20요약.png)

#### Random Sample Heatmap: GT vs Pred
![랜덤 샘플 히트맵](fig/랜덤%20샘플%20히트맵:%20GT%20vs%20Pred.png)

#### Distribution Overlay: GT vs Pred (Presence)
![분포 오버레이 Presence](fig/유전자별%20분포%20오버레이%20(GT%20vs%20Pred)(Presence).png)

#### Distribution Overlay: GT vs Pred (Expression)
![분포 오버레이 Expression](fig/유전자별%20분포%20오버레이%20(GT%20vs%20Pred)(Expression).png)

#### GT Mean vs Pred Mean (Presence / Expression / Uncertainty)
![전체 평균 비교](fig/전체%20평균%20비교:%20GT%20Mean%20vs%20Pred%20Mean.png)

#### Individual Sample Comparison (Presence / Expression / Uncertainty)
![개별 샘플 비교](fig/개별%20샘플%20막대%20차트:%20GT%20vs%20Pred.png)

#### Patch Image + GT vs Pred (5-column)
![패치 이미지 비교](fig/패치%20이미지%20+%20GT%20vs%20Pred%20비교%20시각화.png)

#### Global PCC Analysis (3×3 subplot)
![Global PCC Analysis](fig/Global%20PCC%20Analysis.png)

---

## 4. Discussion

### Presence Head
- 전체 유전자 평균 Accuracy **0.9103**으로 높은 binary classification 성능 확인
- Endothelial (PECAM1, 0.9409), Stromal 계열 (0.9360)에서 가장 높은 정확도
- B Cell 계열 (CD79A, MS4A1, BANK1, TCL1A)에서 상대적으로 낮은 정확도 (0.8359)
  - 특히 TCL1A (0.7787)는 germinal center B cell 특이적으로 형태학적 구분이 어려움
- PCC 기준 평균 **0.7338** — binary label 특성상 연속 PCC보다는 accuracy가 더 적합한 지표

### Expression Head (Conditional)
- 평균 PCC **0.9424**로 매우 높은 조건부 발현량 예측 성능 달성
- 전체 19개 유전자에서 PCC > 0.91 (최저: SFRP4 0.9129, MRC1 0.9123)
- SmoothL1 + PCC loss 조합으로 안정적 학습 및 상관 패턴 최적화
- Zero-inflated 문제를 Presence head로 분리하여 regression 성능 대폭 향상
- 평균 MAE **0.0484**로 낮은 절대 오차 유지

### Combined Prediction (P × μ)
- Final prediction = sigmoid(presence_logits) × expression_mu
- 평균 PCC **0.9419**, Global PCC **0.9527**
- Presence와 Expression의 곱으로 zero-inflated distribution을 자연스럽게 모델링
- 미발현 gene의 prediction을 0에 가깝게 억제하여 MAE **0.0470** 달성

### Uncertainty Estimation
- 평균 σ **0.0530** — 전체적으로 낮은 불확실성으로 모델의 높은 신뢰도 시사
- SFRP4 (σ=0.0734), CD79A (σ=0.0668) 등 실제 예측 난이도가 높은 유전자에서 높은 σ 학습
- AIF1 (σ=0.0416), CD4 (σ=0.0401) 등 예측 성능이 좋은 유전자에서 낮은 σ
- σ-Error PCC **0.1865** — uncertainty가 실제 오차와 양의 상관관계를 보이나 calibration 개선 여지 있음

### Gene Group Analysis
- **Epithelial/Tumor** (PCC_Comb 0.9482): 형태학적으로 뚜렷한 특징 → 높은 예측 성능
- **Macrophage/Myeloid** (PCC_Comb 0.9432): immune cell infiltration 패턴이 조직 morphology에 잘 반영됨
- **T Cell** (PCC_Comb 0.9431): CD4 (0.9585)가 가장 높고 CD8A (0.9292)가 가장 낮음
- **B Cell** (PCC_Comb 0.9349, Pres_Acc 0.8359): 조직 내 산발적 분포로 가장 낮은 성능
  - TCL1A (Pres_Acc 0.7787)가 전체 유전자 중 가장 낮음

### Multi-Scale Input 효과
- 20x (세포 수준) + 5x (조직 구조 수준) dual-encoder 구조 채택
- 세포 형태와 조직 맥락 정보를 동시에 활용하여 예측 성능 향상

### 한계 및 향후 과제
- Uncertainty calibration 개선 필요 (σ-Error PCC 0.1865 → 목표 0.4+)
- B Cell 특히 TCL1A presence 예측 개선 (형태학적 특징 보강)
- Transformer 기반 encoder 또는 pathology foundation model 적용 검토
- Gene pathway 수준 예측으로 확장 가능성
- Multi-task learning (cell type + gene expression) 통합 학습
- WSI-level inference pipeline 구축 및 spatial map 시각화

---

## 5. Project Structure

```
marker_gene_spatial_expression_prediction/
├── .claude/
│   └── CLAUDE.md
├── fig/                              # 평가 결과 시각화
├── xenium_data_preprocessing.ipynb   # Xenium 데이터 전처리
├── xenium_train_2head.ipynb          # Two-Head 모델 학습 (legacy)
├── xenium_train_3head.ipynb          # Three-Head 모델 학습 (current)
├── xenium_test.ipynb                 # 모델 평가 및 시각화
├── xenium_all_genes.csv              # Xenium 전체 유전자 목록
└── README.md
```

## 6. Technology Stack

| Category | Detail |
|---|---|
| Language | Python |
| Framework | PyTorch |
| Encoder | ConvNeXt-Tiny (timm via segmentation_models_pytorch) |
| Environment | CUDA GPU, Jupyter Notebook |
| Key Libraries | numpy, pandas, torch, torchvision, segmentation_models_pytorch, scikit-learn, scipy, matplotlib, PIL, OpenCV |

<!-- TODO: 아키텍처 전체 다이어그램 이미지 삽입 -->
