# MNIST Low-Data Binary Experiments (Digits 0 vs 1)

## Date
Week 2 – Low-data experiment phase

## Objective
To evaluate how different learning models behave as the amount of training data is reduced, using a controlled binary classification task on MNIST (digits 0 vs 1).

The goal of this experiment is **not to maximize accuracy**, but to observe:
- Data efficiency
- Stability across random seeds
- Performance degradation trends under low-data conditions

This experiment serves as a **baseline diagnostic study** before extending to multi-class MNIST and other datasets.

---

## Dataset Description
- Dataset: MNIST
- Task: Binary classification (digit 0 vs digit 1)
- Image size: 28 × 28 (grayscale)
- Preprocessing:
  - Converted to tensor
  - Pixel values scaled to [0, 1]
  - Images flattened for classical models

### Data Splits
- Test set: Full MNIST test split (filtered to digits 0 and 1)
- Training set: Subsampled fractions of MNIST training split

Training data fractions evaluated:
- 10%
- 25%
- 50%
- 100%

---

## Experimental Setup

### Randomness Control
Each experiment was repeated using three different random seeds:
- 42
- 123
- 999

Random seeds were used to control:
- Training data subsampling
- Model initialization
- DataLoader shuffling

Final results are recorded per seed to allow computation of mean and variance.

---

## Models Evaluated (Current Progress)

### Logistic Regression (Classical Baseline)
- Implementation: scikit-learn `LogisticRegression`
- Input representation: Flattened 784-dimensional vectors
- Optimization: Default L2 regularization, `max_iter=1000`
- Motivation:
  - Serves as a strong linear baseline
  - Known to perform well in low-data regimes
  - Provides a reference ceiling for this task

### Support Vector Machine (SVM + PCA)
- Implementation: scikit-learn `SVC` with RBF kernel
- Feature preprocessing:
  - PCA with 20 components (fit on training data only)
  - Standardization applied after PCA
- Motivation:
  - Strong kernel-based classical baseline
  - Well-known performance in low-data regimes
  - Serves as a classical analogue to quantum kernel methods

---

## Results Summary (Logistic Regression)

Observed test accuracy across data fractions:

- **10% training data**: ~99.85–99.90%
- **25% training data**: ~99.90–99.95%
- **50% training data**: ~99.95%
- **100% training data**: ~99.95%

### Key Observations
1. Logistic regression achieves near-optimal accuracy even with very limited training data.
2. Performance saturates early, indicating that MNIST (0 vs 1) is almost linearly separable.
3. Increasing training data beyond 10–25% provides negligible accuracy gains.
4. Training time scales approximately linearly with dataset size.

These results confirm that **MNIST binary classification is a simple task**, making it useful as a diagnostic benchmark rather than a challenging classification problem.

## Results Summary (SVM + PCA)

Observed test accuracy across data fractions:

- **10% training data**: ~99.80–99.95%
- **25% training data**: ~99.90–99.95%
- **50% training data**: ~99.95–100%
- **100% training data**: ~100%

### Key Observations
1. SVM achieves near-perfect accuracy across all data fractions.
2. Marginal improvements over logistic regression are observed at higher data fractions.
3. Performance variance across random seeds is minimal, indicating high stability.
4. Training time increases with dataset size, reflecting kernel method scalability limits.

---

## Implications for Further Experiments
- Deep models (CNNs) are unlikely to show meaningful gains on this task.
- Any performance gap observed in quantum models should be interpreted in the context of this saturation effect.
- The primary value of this dataset lies in analyzing:
  - Data efficiency trends
  - Model stability
  - Differences in degradation behavior, not peak accuracy

---

## Next Steps
1. Implement SVM + PCA as a stronger classical low-data baseline.
2. Compare logistic regression vs SVM behavior under identical data fractions.
3. Introduce CNN baseline once classical non-deep baselines are established.
4. Integrate quantum models (VQC and quantum kernel) only after classical baselines are finalized.

---

## Notes for Paper Writing
- This experiment motivates the need for more complex datasets (multi-class MNIST, BreastMNIST).
- Results from this section will be used to justify dataset extensions in the methodology and discussion sections.
- Accuracy should be reported alongside variance across seeds to highlight stability differences.
