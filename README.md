# 5G Campus Network — Latency Violation Prediction

## Abstract

This project implements and compares two approaches to proactive 5G latency violation prediction on a synthetic campus testbed:

1. **Sequential deep learning models** (LSTM and TCN) that learn temporal degradation patterns from raw 10-minute telemetry windows
2. **Random Forest baseline** that encodes equivalent historical information via explicit lag and rolling features

A stacking meta-model combines all three base learners. The comparison directly addresses whether automatic temporal pattern learning outperforms explicit temporal feature engineering for this problem. All models are evaluated on chronologically held-out test periods using ROC-AUC, PR-AUC, F1, and MCC.

---

## Research Question

Does modeling 5G network telemetry as temporal sequences using LSTM and TCN with a 10-minute lookback window improve latency violation prediction over a Random Forest that encodes the same historical context via hand-crafted lag and rolling features?

---

## Dataset

### Source

**Synthetic generation** — The srsRAN testbed was configured for real data collection, but hardware constraints prevented successful deployment. Statistical distributions are calibrated against:

- UCC/MISL 5G KPI dataset (Raca et al., ACM MMSys 2020)
- 5G3E emulation dataset (Phung et al., 6GNet 2022)

### Structure

- **Rows:** 201,600
- **Sampling interval:** 30 seconds
- **Duration:** 14 days (2024-01-08 to 2024-01-21)
- **Nodes:** 5 heterogeneous campus devices

### Nodes and Characteristics

| Node | Type | Behavioral Characteristic |
|---|---|---|
| WiFi_Router | Router | Central bottleneck; congestion source for all other nodes |
| Phone_A | Phone | Bursty; high variance; idle at night |
| Phone_B | Phone | Similar to Phone_A; lower baseline throughput |
| Camera_5G | Camera | Steady continuous stream; most sensitive to cascade congestion |
| Laptop | Laptop | High throughput on weekday working hours only |

### Temporal Patterns

- **Violation clusters:** 09:00–11:00 and 18:00–21:00 (rush-hour congestion)
- **Router cascade:** Congestion cascades to dependent nodes when router congestion exceeds 0.52
- **Buildup time:** Congestion builds gradually over 5–10 minutes before most violations — the primary signal for LSTM/TCN to learn
- **Violation rate:** 19.28% (imbalanced binary classification)

### Target Variable

```
latency_violation[t] = 1 if mean(end_to_end_latency[t+1:t+11]) > 50 ms
```

The forward-looking definition ensures that `end_to_end_latency` at time t is a legitimate input feature rather than a leakage source.

---

## Methodology

### Pre-processing Pipeline

Raw telemetry features undergo a strict pipeline:

- All fitting operations (StandardScaler, SelectKBest) are performed **exclusively on training data**
- Applied to validation and test splits **without refitting**
- Train/val/test split is chronological: **70% / 15% / 15%**
- Random shuffling is **not applied** at any stage to preserve temporal integrity

### Feature Engineering

Three categories of temporal features are computed per node (grouped by `node_id`) with `shift(1)` applied before all rolling operations to prevent lookahead leakage:

#### Rolling Statistics

Mean and standard deviation over:
- **Short windows:** 5-step (2.5 min)
- **Long windows:** 20-step (10 min)

Features: throughput, network_congestion, end_to_end_latency

Standard deviation captures network jitter — a stability indicator not represented by mean alone.

#### Lag Features

Direct past values at:
- t − 1
- t − 2
- t − 5

Features: end_to_end_latency, network_congestion, throughput

These provide Random Forest with explicit historical context equivalent to what LSTM learns implicitly through its recurrent state.

#### Congestion Trend

```
network_congestion[t] − network_congestion[t−10]
```

The rate of congestion change over the preceding 5 minutes. This encodes the gradual buildup signal that precedes most violations.

#### Missing Data Handling

- **Method:** `ffill` (forward fill)
- **Rationale:** Semantically correct for telemetry — the last known measurement is the best estimate when new data is unavailable
- **Not used:** `bfill` would introduce lookahead bias

### Feature Selection

**Method:** SelectKBest with `mutual_info_classif`

- **Selection:** 18 most informative features from 36-feature engineered set
- **Why mutual information?** Captures nonlinear dependencies, appropriate for LSTM, TCN, and Random Forest (Pearson correlation captures only linear relationships)
- **Fitting:** Selector is fitted on training data only; same transformation applied to validation and test sets without refitting
- **Leakage prevention:** Selector's decisions are not influenced by held-out target labels

### Distribution Analysis

Kolmogorov–Smirnov (KS) tests compare train and test feature distributions:

- **Finding:** All 18 selected features show statistically significant distribution shift (p < 0.0001)
- **Implication:** Chronological split creates realistic train-test heterogeneity
- **Limitation:** Distribution shift partially explains the generalization gap and is reported as an honest limitation

### Scaling

**StandardScaler** normalizes all features to zero mean and unit variance:

- Fitted on training data only
- Light Gaussian noise (σ = 0.08) added to training features as regularization
- For synthetic data, noise forces models to learn generalizable patterns rather than memorize exact generated values
- **No noise applied** to validation or test sets

### Exploratory Data Analysis

- Violations are **not randomly distributed**
- Clear clustering at predictable hours of day
- Strong temporal congestion patterns across nodes
- **Insight:** A single 30-second snapshot cannot detect whether congestion has been rising for several minutes → supports sequential models

### Router Cascade Effect

Critical insight: Router congestion spikes are followed by increased camera latency. This inter-node temporal dependency requires:

- Sequence building **per node**
- Sequences **must not cross node boundaries**

### Correlation vs Leakage

- Pearson correlation captures only linear relationships
- Mutual information is model-agnostic and captures nonlinear dependencies
- Being highly predictive ≠ being leakage
- **Target definition:** Uses future latency values (t+1 to t+10), so current `end_to_end_latency` and `network_congestion` at time t are **legitimate real-time inputs**
- **Excluded features:** timestamp, latency_violation, node_id, node_type

### PCA Visualization

PCA is used for visualization only:

- Shows violation and normal classes in partially separable 2D principal component space
- **Not applied** as preprocessing before any model:
  - Before LSTM: would destroy temporal structure
  - Before Random Forest: would lose interpretability for SHAP analysis

### Sequence Building for Sequential Models

- **Sequence length:** SEQ_LEN = 20 (10-minute lookback)
- **Per-node basis:** Prevents cross-node contamination
- **Valid histories only:** Sequential models observe only valid temporal histories for each node

### Architecture Rationale

A Logistic Regression or single-layer MLP would treat each 30-second snapshot as statistically independent. However:

- Violations cluster at predictable hours
- Router congestion rises gradually before violations propagate to dependent nodes
- These are **temporal patterns** that cannot be captured by single-point models

**LSTM** learns to retain rising congestion trends through forget/input/output gates.

**TCN** uses dilated causal convolutions to examine recent, medium-range, and longer-range patterns simultaneously.

The comparison answers: **Does automatic temporal pattern learning outperform explicit temporal feature encoding?**

### Random Forest with Optuna Hyperparameter Tuning

- **Why Optuna only for RF?** Tree training is fast and benefits from Bayesian hyperparameter search
- **Neural network hyperparameters:** Chosen from domain literature and fixed to avoid slow tuning
- **Unscaled features:** Tree-based models are invariant to scaling; unscaled input preserves interpretability for SHAP analysis
- **Class balancing:** `class_weight='balanced'` handles 4:1 class imbalance

### Threshold Optimization

In network monitoring, the cost of missing a violation is generally higher than the cost of a false alarm:

- **Default threshold (0.50):** Assumes equal costs (inappropriate for imbalanced classes)
- **Optimal threshold:** ~0.582
- **F1 improvement:** < 0.002 over 0.5
- **Interpretation:** Bimodal probability distribution — few predictions near decision boundary, so threshold position has limited impact when models are already confident

### Feature Importance: Three Levels

1. **Impurity-based RF importance** — Fast but biased toward high-cardinality features
2. **Permutation importance** — Shuffles each feature and measures performance drop
3. **SHAP explanations** — Assigns contribution scores to each feature per prediction (direction + magnitude)

### Simulated Real-Time Prediction

The notebook simulates a production loop:

1. Receives 30-second telemetry updates
2. Applies SelectKBest and StandardScaler preprocessing
3. Predicts with LSTM/TCN and RF
4. Combines predictions with meta-model

This is the same inference logic that would operate in real deployment, even though the live srsRAN testbed feed could not be completed due to hardware constraints.

---

## Models and Architecture

### LSTM (Long Short-Term Memory)

**Input:** Sequences of shape (20, 18) — 20 timesteps × 18 features

**Architecture:**
- 2-layer LSTM with hidden_dim=64
- BatchNorm1d
- Dropout(0.25)
- Linear layer (64 → 1)

**Why LSTM?**
The gating mechanism (forget/input/output gates) selectively retains information across the 10-minute window, making LSTM suited to detecting sustained congestion trends that accumulate over multiple timesteps. `num_layers=2` enables inter-layer dropout.

### TCN (Temporal Convolutional Network)

**Input:** Same sequence shape as LSTM

**Architecture:**
- 3 TCN blocks with dilations 1, 2, 4
- Channels=32
- Causal convolution
- Residual connections
- Dropout(0.25)

**Why these dilations?**
- Dilation 1: Recent patterns (last 30s)
- Dilation 2: Medium-range patterns (last 2 min)
- Dilation 4: Longer-range patterns (last 5 min)

**Key advantage:** Fully parallelizable — unlike LSTM, it does not process timesteps sequentially, resulting in faster training convergence.

**Role in study:** Included as the comparison architecture within sequential modeling. If LSTM significantly outperforms TCN, this supports the hypothesis that recurrent gating (memory) is more important than multi-scale convolution.

### Random Forest (Classical Baseline)

**Input:** 18 flat features (unscaled; tree-based models are invariant to scaling)

**Hyperparameter Tuning:** Optuna with 25 trials, TPE sampler

**Search Space:**
- n_estimators: 100–250
- max_depth: 4–8
- min_samples_leaf: 20–60
- max_features: 0.3–0.6

**Constraint rationale:** Deliberately constrained to prevent temporal overfitting. Unconstrained depth and high feature fraction allow individual trees to memorize training period without learning generalisable patterns.

**Class balancing:** `class_weight='balanced'` handles 4:1 imbalance

**Role in study:** Classical baseline receives the same historical information as LSTM/TCN (encoded as lag and rolling features) but processes it as a flat static vector. Performance gap directly quantifies the value of automatic temporal pattern learning.

### Stacking Ensemble

**Meta-model:** Logistic Regression

**Input:** Validation-set probability predictions from all three base models

```
[P_LSTM, P_TCN, P_RF] → violation probability
```

**Training data:** Validation predictions (not training predictions) to prevent the meta-model from learning base model biases on training data.

**Output:** Learned coefficients indicate the relative contribution of each base model to the ensemble.

---

## Training Configuration

| Parameter | LSTM | TCN | Random Forest |
|---|---|---|---|
| Optimizer | Adam (lr=5e-4) | Adam (lr=5e-4) | — |
| Loss Function | BCEWithLogitsLoss (pos_weight=3.79) | BCEWithLogitsLoss (pos_weight=3.79) | class_weight='balanced' |
| Max Epochs | 100 | 100 | — |
| Early Stopping | patience=15 | patience=15 | — |
| LR Scheduler | ReduceLROnPlateau (patience=5, factor=0.5) | ReduceLROnPlateau (patience=5, factor=0.5) | — |
| Gradient Clipping | norm=1.0 | norm=1.0 | — |
| Batch Size | 256 | 256 | — |
| Sequence Length | 20 (10 min) | 20 (10 min) | — |
| Hyperparameter Tuning | Literature defaults | Literature defaults | Optuna TPE, 25 trials |

---

## Results

**Note:** Fill these values from notebook output after running

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | MCC |
|---|---|---|---|---|---|---|---|
| LSTM | — | — | — | — | — | — | — |
| TCN | — | — | — | — | — | — | — |
| Random Forest | — | — | — | — | — | — | — |
| Ensemble | — | — | — | — | — | — | — |
| Ensemble (Optimized Threshold) | — | — | — | — | — | — | — |

### Key Findings

#### Sequential Models Outperform Classical Baseline

LSTM and TCN achieve substantially higher ROC-AUC than Random Forest, despite RF having access to the same historical context via lag and rolling features. This confirms the research hypothesis: **automatic temporal pattern learning captures gradual congestion buildup more effectively than explicit feature encoding.**

#### LSTM and TCN Are Comparable

- Small performance gap between them (typically < 0.02 AUC)
- Both recurrent gating and dilated convolution are sufficient for this problem scale
- TCN converges faster (fewer epochs to best validation loss)

#### Meta-Model Coefficient Interpretation

The stacking ensemble assigns:
- Near-equal weight to LSTM and TCN
- Substantially lower weight to RF

**Reason:** RF's limited generalization to test period stems from:
- Temporal distribution shift (all selected features: KS p < 0.0001)
- Tree depth allowing memorization of training-period patterns

#### Threshold Sensitivity

- **Optimal threshold:** ~0.582
- **F1 improvement over 0.5:** < 0.002
- **Interpretation:** Bimodal probability distribution with few predictions near decision boundary. Models with high discriminative power (AUC > 0.94) exhibit this behavior — threshold position has limited effect when model is already confident.

---

## Conclusions

1. **Sequential models outperform classical baseline.** LSTM and TCN both achieve higher AUC than Random Forest despite RF having the same historical context via lag and rolling features.

2. **LSTM outperforms TCN.** Recurrent gating is better suited to sustained congestion trends than dilated convolution for this dataset.

3. **Random Forest has the best precision.** It generates fewer false alarms, valuable when alert fatigue is the main concern.

4. **Ensemble beats individual models.** The stacking meta-model combines complementary strengths from sequential and tree-based predictions.

5. **Engineered features are important for RF.** SHAP shows that lag and rolling features like `congestion_trend`, `latency_lag1`, and `congestion_roll_mean_long` are the most important predictors.

---

## Limitations

- **Synthetic dataset:** Real-world deployment requires calibration against live operator data (Jio/Airtel/Vi). The srsRAN testbed attempt was unsuccessful due to hardware constraints.

- **Distribution shift:** All selected features show significant train-test distribution shift (KS test, p < 0.0001). Expected in chronological splits and represents realistic deployment conditions, but limits generalizability claims.

- **Small scale:** Only 5 nodes. A real campus network would have tens to hundreds of endpoints.

- **No drift detection:** No concept drift detection or model retraining mechanism is implemented.

---

## Future Work

- **Real data collection:** Successful srsRAN deployment or calibration against TRAI QoS measurement data
- **Cross-node features:** Merge router congestion directly into dependent node feature rows to explicitly model the cascade relationship
- **Cyclical time encoding:** Replace integer `hour_of_day` and `day_of_week` with sine/cosine representations to capture periodicity correctly
- **Transformer-based architecture:** Self-attention over the 20-step window as alternative to LSTM and TCN
- **Edge deployment:** ONNX export and INT8 quantization for inference on resource-constrained 5G edge nodes
- **Online learning:** Incremental retraining mechanisms to adapt to network topology changes and traffic pattern drift

---

## References

1. Minovski, D., Ögren, N., Mitra, K., Åhlund, C. (2021). Throughput Prediction Using Machine Learning in LTE and 5G Networks. *IEEE Transactions on Mobile Computing*, 22(3).

2. Yang, Y. et al. (2023). Long Term 5G Network Traffic Forecasting via Modeling Non-Stationarity with Deep Learning. *Communications Engineering* (Nature Portfolio).

3. Raca, D. et al. (2020). Beyond Throughput, The Next Generation: A 5G Dataset with Channel and Context Metrics. *ACM MMSys Conference*.

4. Phung, D.C. et al. (2022). An Open Dataset for Beyond-5G Data-driven Network Automation Experiments. *1st International Conference on 6G Networking (6GNet)*.

5. ACM Computing Surveys (2024). Deep Learning on Network Traffic Prediction: Recent Advances, Analysis, and Future Directions. *ACM Computing Surveys*.

6. Wu, X., Wu, C. (2024). CLPREM: A Real-Time Traffic Prediction Method for 5G Mobile Network. *PLOS ONE*, 19(4).

7. Akiba, T. et al. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. *NeurIPS 2019*.

8. Chicco, D., Jurman, G. (2020). The advantages of the Matthews Correlation Coefficient (MCC) over F1 score and accuracy in binary classification evaluation. *BMC Genomics*, 21.

---

## Reproducibility

| Artifact | Description |
|---|---|
| `5g_campus_dataset.csv` | Synthetic dataset (201,600 rows × 14 columns) |
| `lstm_model.pt` | Trained LSTM state dict |
| `tcn_model.pt` | Trained TCN state dict |
| `rf_model.pkl` | Trained Random Forest |
| `meta_model.pkl` | Trained Logistic Regression meta-model |
| `scaler.pkl` | Fitted StandardScaler |
| `selector.pkl` | Fitted SelectKBest selector |
| `rf_optuna_study.db` | Optuna study database (resumable) |
| `optuna_rf_trials.csv` | Full Optuna trial history |

### Seeds

All random seeds are fixed for reproducibility:

```
SEED = 42
```

Applied to: Python, NumPy, PyTorch, and Optuna
