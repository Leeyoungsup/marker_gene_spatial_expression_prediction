# Marker Gene Spatial Expression Prediction

H&E 염색 병리 이미지로부터 marker gene의 spatial expression을 예측하는 딥러닝 모델

---

## 1. Introduction

### Background
- Spatial transcriptomics 데이터는 비용과 실험 제약으로 대규모 확보가 어려움
- 조직의 morphology는 cell type composition, tumor microenvironment, immune infiltration 등과 강한 상관관계를 가짐
- H&E 이미지의 형태학적 특징만으로 유전자 발현의 공간적 분포를 추정할 수 있는 가능성 존재

### Objective
- **H&E patch → gene expression vector** 형태의 regression 모델 학습
- Spot-level에서 19개 marker gene의 **proportion (양성 비율)** 및 **intensity (발현 강도)** 동시 예측
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

---

## 2. Method

### 2.1 Data

- **데이터 소스**: Xenium spatial transcriptomics
- **WSI 분할**: 20x 기준 512×512 non-overlapping grid
- **패치 포함 조건**: 패치 내 ST spot ≥ 1개
- **Train/Val split**: 80/20 (random_state=42)

### 2.2 Model Input (Multi-Scale Patch)

| Scale | Resolution | Shape | Description |
|---|---|---|---|
| High-resolution | 20x | (3, 512, 512) | 세포 morphology 정보 |
| Context | 5x | (3, 512, 512) | 넓은 조직 구조 정보 (동일 중심, ~4배 넓은 영역) |

### 2.3 Model Architecture

- **Encoder**: ConvNeXt-Tiny (`tu-convnext_tiny`, ImageNet pretrained) × 2 (20x, 5x 각각)
- **Feature Fusion**: 두 encoder의 last-stage feature를 element-wise addition
- **Pooling**: Global Average Pooling → (B, 768)
- **Two-Head 구조**:
  - **Head A (Proportion)**: Linear(768→256) → ReLU → Dropout(0.2) → Linear(256→19) → Sigmoid (inference 시)
  - **Head B (Intensity)**: Linear(768→256) → ReLU → Dropout(0.2) → Linear(256→19) → Sigmoid

### 2.4 Label Generation

- **Head A — Proportion** (양성 비율, range [0, 1])
  ```
  proportion[g] = (패치 내 gene g 발현 spot 수) / (패치 내 전체 spot 수)
  ```
- **Head B — Intensity** (발현 강도, range [0, 1])
  ```
  intensity[g] = clip(mean(log1p(raw_count[양성 spot])), 0, 10) / 10
  ```
  - Xenium은 imaging 기반 기술로, 시퀀싱 depth bias가 없어 CPM 정규화 불필요
  - Raw transcript count에 log1p 변환 후 [0, 10] clip → 10으로 나누어 [0, 1] 범위로 정규화

### 2.5 Loss Function

```
Total Loss = α × BCE(prop_logits, gt_prop) + β × MSE(pred_int, gt_int) + γ × PCC_loss
```

| Component | Function | Weight |
|---|---|---|
| Proportion Loss | BCEWithLogitsLoss | α = 1.0 |
| Intensity Loss | MSELoss | β = 1.0 |
| PCC Loss | 1 − mean(Pearson Correlation) | γ = 0.5 |

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

---

## 3. Result

### 3.1 Per-Gene Performance

| Gene | PCC (Proportion) | MAE (Proportion) | PCC (Intensity) | MAE (Intensity) |
|---|---|---|---|---|
| EPCAM | 0.9176 | 0.0956 | 0.7883 | 0.0686 |
| EGFR | 0.9610 | 0.0709 | 0.6874 | 0.0577 |
| ACTA2 | 0.9322 | 0.0755 | 0.7611 | 0.0772 |
| PDGFRA | 0.9499 | 0.0766 | 0.6849 | 0.0717 |
| PDGFRB | 0.9535 | 0.0676 | 0.7715 | 0.0601 |
| SFRP4 | 0.9304 | 0.0968 | 0.7506 | 0.0815 |
| PECAM1 | 0.9463 | 0.0708 | 0.7162 | 0.0720 |
| CD68 | 0.9565 | 0.0612 | 0.7651 | 0.0647 |
| AIF1 | 0.9462 | 0.0690 | 0.6925 | 0.0701 |
| FCGR3A | 0.9625 | 0.0634 | 0.7326 | 0.0640 |
| MRC1 | 0.9287 | 0.1010 | 0.7308 | 0.0621 |
| CD3E | 0.9047 | 0.1177 | 0.6625 | 0.0781 |
| CD4 | 0.9449 | 0.0702 | 0.7148 | 0.0658 |
| CD8A | 0.9305 | 0.1045 | 0.6537 | 0.0667 |
| TRAC | 0.9097 | 0.1084 | 0.7196 | 0.0777 |
| CD79A | 0.8821 | 0.1129 | 0.7616 | 0.0671 |
| MS4A1 | 0.8751 | 0.1204 | 0.7848 | 0.0626 |
| BANK1 | 0.8814 | 0.1254 | 0.7460 | 0.0618 |
| TCL1A | 0.8403 | 0.1211 | 0.7410 | 0.0616 |
| **Mean** | **0.9239** | **0.0910** | **0.7297** | **0.0679** |

### 3.2 Summary

- **Proportion Head**: 평균 PCC **0.9239**, 평균 MAE **0.0910**
- **Intensity Head**: 평균 PCC **0.7297**, 평균 MAE **0.0679**
- 전체 19개 유전자에서 Proportion PCC > 0.84, Intensity PCC > 0.65 달성

### 3.3 Gene Group Performance

| Gene Group | Proportion PCC | Intensity PCC |
|---|---|---|
| Epithelial (EPCAM, EGFR) | ~0.94 | ~0.74 |
| Stromal (ACTA2, PDGFRA, PDGFRB, SFRP4) | ~0.94 | ~0.74 |
| Endothelial (PECAM1) | 0.95 | 0.72 |
| Macrophage (CD68, AIF1, FCGR3A, MRC1) | ~0.95 | ~0.73 |
| T Cell (CD3E, CD4, CD8A, TRAC) | ~0.92 | ~0.69 |
| B Cell (CD79A, MS4A1, BANK1, TCL1A) | ~0.87 | ~0.76 |

### 3.4 Figures

#### Per-Gene PCC & MAE Bar Chart
![유전자별 PCC & MAE 막대 차트](fig/유전자별%20PCC%20&%20MAE%20막대%20차트.png)

#### Scatter Plot: Pred vs GT (Proportion)
![Scatter Plot Proportion](fig/Scatter%20Plot:%20Pred%20vs%20GT(proportion).png)

#### Scatter Plot: Pred vs GT (Intensity)
![Scatter Plot Intensity](fig/Scatter%20Plot:%20Pred%20vs%20GT(Intensity).png)

#### Error Distribution Boxplot
![오차 분포 Boxplot](fig/오차%20분포%20Boxplot.png)

#### Gene Group Performance Summary
![유전자 그룹별 성능 요약](fig/유전자%20그룹별%20성능%20요약.png)

#### Random Sample Heatmap: GT vs Pred
![랜덤 샘플 히트맵](fig/랜덤%20샘플%20히트맵:%20GT%20vs%20Pred.png)

#### Distribution Overlay: GT vs Pred (Proportion)
![분포 오버레이 Proportion](fig/유전자별%20분포%20오버레이%20(GT%20vs%20Pred)(Proportion).png)

#### Distribution Overlay: GT vs Pred (Intensity)
![분포 오버레이 Intensity](fig/유전자별%20분포%20오버레이%20(GT%20vs%20Pred)(Intensity).png)

#### GT Mean vs Pred Mean
![전체 평균 비교](fig/전체%20평균%20비교:%20GT%20Mean%20vs%20Pred%20Mean.png)

#### Individual Sample Comparison
![개별 샘플 막대 차트](fig/개별%20샘플%20막대%20차트:%20GT%20vs%20Pred.png)

#### Patch Image + GT vs Pred
![패치 이미지 비교](fig/패치%20이미지%20+%20GT%20vs%20Pred%20비교%20시각화.png)

#### Global PCC Analysis
![Global PCC Analysis](fig/Global%20PCC%20Analysis.png)

---

## 4. Discussion

### Proportion Head
- 전체 유전자 평균 PCC 0.92로 높은 예측 성능 확인
- Macrophage/Myeloid 계열 (CD68, FCGR3A) 유전자에서 가장 높은 PCC (>0.96) 달성
- B Cell 계열 (CD79A, MS4A1, BANK1, TCL1A)에서 상대적으로 낮은 PCC (~0.87)
  - B cell은 조직 내 분포가 산발적이며 형태학적 특징이 덜 두드러짐
- MAE는 전체적으로 0.06~0.13 범위로 안정적

### Intensity Head
- 평균 PCC 0.73으로 Proportion 대비 낮은 성능
- 발현 강도는 형태학적 특징만으로 설명하기 어려운 요소가 포함됨
  - transcriptional regulation, epigenetic state 등 비형태학적 요인 존재
- B Cell 계열에서 Intensity PCC가 Proportion 대비 상대적으로 양호 (~0.76)
- T Cell 계열 (CD3E, CD8A)에서 Intensity 예측이 가장 어려움 (PCC ~0.65)

### Multi-Scale Input 효과
- 20x (세포 수준) + 5x (조직 구조 수준) dual-encoder 구조 채택
- 세포 형태와 조직 맥락 정보를 동시에 활용하여 예측 성능 향상

### 한계 및 향후 과제
- Intensity 예측 성능 개선 필요 (PCC 0.73 → 목표 0.80+)
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
├── xenium_train_2head.ipynb          # Two-Head 모델 학습
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
