# Day 4 Sprint Plan: Model Improvement with Cost Optimization

**Date**: 2026-06-25  
**Compute Budget**: ~6-8% of purchased quota (based on Day 3 usage: 6% for training + analysis)  
**Goal**: Improve baseline model while managing cloud compute costs  
**Sprint Container**: Issue #13 (Day 4 Sprint)  

## Context: Baseline Analysis Findings

From BASELINE_ANALYSIS.md:
- **Current Model**: U-Net, 31.1M params, trained 20 epochs
- **Test Performance**: Accuracy 89.2%, Precision 94.2%, Recall 60.5%, IoU 0.584, ROC AUC 0.920
- **Key Limitation**: Conservative model (high precision, moderate recall) — misses 40% of burned pixels
- **Improvement Opportunities**:
  1. Class weighting in loss (expect +5-10% recall)
  2. Loss function tuning (30% BCE + 70% Dice vs current 50/50)
  3. Extended training (30-50 epochs with early stopping)
  4. Learning rate adjustment
  5. Cross-validation for robustness

## Compute Cost Analysis

**Day 3 Baseline:**
- Training: 20 epochs on T4 GPU = ~20 minutes = ~0.33 GPU hours
- Inference: 644 test samples on T4 GPU = ~10 minutes = ~0.17 GPU hours
- Analysis: sklearn on CPU = ~5 minutes = ~0.08 CPU hours
- **Total**: ~0.58 GPU hours (6% of monthly quota)

**Day 4 Budget Strategy:**
- **Compute Available**: ~8 GPU hours (remaining monthly quota comfortably)
- **Target**: Run 3-4 major experiments without exceeding ~12% total usage
- **Per Experiment**: 1-2 hours GPU time

| Experiment | Duration | Cost | Expected Gain |
|------------|----------|------|---|
| Class weighting (20 epochs) | 20 min | 0.33 hrs | +5-10% recall |
| Extended training (50 epochs) | 50 min | 0.83 hrs | +2-5% IoU |
| Loss tuning (30% BCE / 70% Dice) | 50 min | 0.83 hrs | +1-3% IoU |
| Learning rate search (3 seeds) | 60 min | 1.0 hrs | +1-2% IoU |
| **Subtotal** | — | **3.0 hrs** | **+8-20% recall, +3-8% IoU** |

**Note**: Cross-validation (#16) deferred to Day 5 (expensive: 5× training time)

---

## Sprint Structure

### Phase 1: Quick Wins (High ROI, Low Cost)
**Duration**: ~1 hour GPU  
**Goal**: Immediate improvements with minimal compute spend

### Phase 2: Main Experiments (Controlled Investigation)
**Duration**: ~2 hours GPU  
**Goal**: Test hypotheses from baseline analysis

### Phase 3: Decision Point
**Gate**: Review results, decide if additional experiments justified

### Phase 4: Analysis & Documentation
**Duration**: Async (CPU/IDE)  
**Goal**: Document results, update Issue #20

---

## Issue #14: Hyperparameter Tuning

**Objective**: Improve recall and IoU through parameter adjustment  
**Expected Impact**: +5-10% recall, +2-5% IoU  
**Estimated Cost**: 1.5 GPU hours

### Task Checklist

#### 1. Hypothesis: Class Weighting (HIGH PRIORITY - Quick Win)
- [ ] **Setup** (5 min)
  - [ ] Open 03_training.ipynb on Colab (GPU runtime)
  - [ ] Backup current notebook version locally
  - [ ] Document baseline checkpoint path: `/content/drive/MyDrive/RETINNA_checkpoints/best_model.pth`

- [ ] **Implement** (10 min)
  - [ ] Modify `BCEDiceLoss` to accept class weights
  - [ ] Calculate class weights based on burn prevalence (25% burned, 75% unburned)
    ```python
    # Burned class: weight=0.6, Unburned class: weight=0.4
    # (favors learning burned class)
    burned_weight = 0.6
    unburned_weight = 0.4
    ```
  - [ ] Add weight parameter to loss initialization in training cell

- [ ] **Train** (20 min GPU)
  - [ ] Train for 20 epochs (same as baseline for fair comparison)
  - [ ] Monitor: Compare training loss, validation loss, validation IoU vs baseline
  - [ ] **Success Criterion**: Recall should improve to 62-65% (vs baseline 60.5%)
  - [ ] Save checkpoint to: `checkpoints_notebook/best_model_weighted.pth`
  - [ ] Backup to Drive: `/content/drive/MyDive/RETINNA_checkpoints/best_model_weighted_epoch{N}.pth`

- [ ] **Quick Eval** (5 min)
  - [ ] Load checkpoint in same notebook
  - [ ] Run inference on 10 test samples (quick visual check)
  - [ ] Compare prediction confidence vs baseline
  - [ ] Document: "Recall improved from 60.5% to X%, precision slight drop from 94.2% to Y%"

- [ ] **Decision Gate**:
  - [ ] If recall improved >5%: Proceed to Issue #15
  - [ ] If recall improved <2%: Try higher weight ratio (0.7/0.3)
  - [ ] Document reasoning and move forward

#### 2. Hypothesis: Extended Training (MEDIUM PRIORITY)
- [ ] **Setup** (5 min)
  - [ ] Create new checkpoint directory: `checkpoints_notebook/best_model_extended/`
  - [ ] Set epochs slider to 50 (in 03_training.ipynb)

- [ ] **Train** (50 min GPU)
  - [ ] Train on class-weighted loss (from Task 1 if successful)
  - [ ] Monitor validation IoU for early stopping pattern
  - [ ] Log: epoch number where val_iou peaks
  - [ ] **Success Criterion**: Best val IoU should reach 0.55+ (vs baseline 0.5201)
  - [ ] Save best checkpoint

- [ ] **Eval** (10 min)
  - [ ] Load best model from extended training
  - [ ] Run full test set inference
  - [ ] Compare: IoU, precision, recall vs baseline
  - [ ] Document: "Extended training improved IoU from 0.584 to X%"

- [ ] **Decision Gate**:
  - [ ] If IoU improved >2%: Keep for final evaluation
  - [ ] If IoU improved <1%: Recommend 20-30 epochs sweet spot
  - [ ] Move to Issue #15

#### 3. Hypothesis: Learning Rate Search (LOW PRIORITY - If Time Permits)
- [ ] **Setup** (5 min)
  - [ ] Identify baseline learning rate: 0.0005
  - [ ] Plan search: [0.0001, 0.0005, 0.001]

- [ ] **Ablation** (30 min GPU for all 3 seeds)
  - [ ] Run 20 epochs with each learning rate
  - [ ] Quick evaluation on validation set only
  - [ ] Compare convergence speed and best IoU
  - [ ] **Success Criterion**: Find learning rate with 1-2% higher IoU

- [ ] **Analysis**:
  - [ ] Document: "Optimal learning rate: 0.0005 (confirmed) / 0.0001 / 0.001"
  - [ ] Recommend for Issue #15 experiments

### Issue #14 Success Metrics

| Goal | Baseline | Target | Status |
|------|----------|--------|--------|
| Recall | 60.5% | 65-70% | TBD |
| IoU | 0.584 | 0.61+ | TBD |
| Precision | 94.2% | ≥93% | TBD |
| Training Epochs | 20 | 30-50 | TBD |

---

## Issue #15: Loss Optimization

**Objective**: Improve IoU and F1-score through loss function tuning  
**Expected Impact**: +2-5% IoU, +3-5% F1  
**Estimated Cost**: 1.5 GPU hours

### Task Checklist

#### 1. Current Loss Function Analysis
- [ ] **Document** (5 min)
  - [ ] Current: 50% BCE + 50% Dice
  - [ ] Rationale: Balance per-pixel accuracy with IoU
  - [ ] Issue: Dice loss may not weight classes equally when class imbalanced

- [ ] **Hypothesis**: Increase Dice weight (favor IoU over pixel accuracy)
  - [ ] Theory: Burned class is sparse (25%), Dice handles this better
  - [ ] Expected effect: IoU improves, precision may slightly decrease

#### 2. Loss Ratio Experiments (3 Seeds)
- [ ] **Experiment 1: 30% BCE + 70% Dice** (20 min GPU)
  - [ ] Modify loss in 03_training.ipynb:
    ```python
    return 0.3 * bce_loss + 0.7 * dice_loss
    ```
  - [ ] Train for 20 epochs (class-weighted from Issue #14 if successful)
  - [ ] Checkpoint: `best_model_loss_30_70.pth`
  - [ ] Record: Best validation IoU, epoch number

- [ ] **Experiment 2: 20% BCE + 80% Dice** (20 min GPU)
  - [ ] Same setup, different ratio
  - [ ] Checkpoint: `best_model_loss_20_80.pth`
  - [ ] Compare to Experiment 1

- [ ] **Experiment 3: 40% BCE + 60% Dice** (20 min GPU)
  - [ ] Conservative approach (smaller step from baseline)
  - [ ] Checkpoint: `best_model_loss_40_60.pth`
  - [ ] Compare to baseline (50/50)

#### 3. Comparative Analysis
- [ ] **Test Set Evaluation** (15 min, multi-threaded inference)
  - [ ] Load each loss-tuned checkpoint
  - [ ] Run inference on full test set
  - [ ] Record metrics table:
    ```
    Loss Ratio | IoU    | Precision | Recall | F1-Score | Recommendation
    50/50 (BL) | 0.5836 | 0.942     | 0.605  | 0.737    | Baseline
    30/70      | ?      | ?         | ?      | ?        | ?
    40/60      | ?      | ?         | ?      | ?        | ?
    20/80      | ?      | ?         | ?      | ?        | ?
    ```

- [ ] **Decision Matrix**:
  - [ ] Best IoU > baseline (0.5836)? YES → Use new ratio
  - [ ] Recall improved? (secondary goal) YES → Bonus
  - [ ] Precision dropped >2%? NO → Acceptable
  - [ ] **Recommendation**: Select best performing ratio

#### 4. Optimal Configuration Selection
- [ ] **Gate Decision**: Which loss ratio to carry forward?
  - [ ] Option A: Best IoU ratio (primary metric)
  - [ ] Option B: Best F1-score ratio (balanced)
  - [ ] Option C: Best recall ratio (if Issue #14 didn't improve enough)
  - [ ] Document reasoning and save checkpoint as: `best_model_final_loss.pth`

### Issue #15 Success Metrics

| Goal | Baseline | Target | Status |
|------|----------|--------|--------|
| IoU | 0.584 | 0.60+ | TBD |
| F1-Score | 0.737 | 0.75+ | TBD |
| Precision | 94.2% | ≥92% | TBD |
| Optimal Ratio | 50/50 | TBD | TBD |

---

## Issue #16: Cross-Validation (DEFERRED)

**Status**: Deferred to Day 5 or later sprint  
**Reason**: Requires 5× training compute (5 folds × 20-50 epochs each)  
**Estimated Cost**: 4-5 GPU hours (16-20% of monthly quota)

### Placeholder Checklist (For Reference)

- [ ] **Setup**: Implement k-fold cross-validation wrapper
- [ ] **Train**: Run 5 folds with best configuration from Issues #14-15
- [ ] **Eval**: Compute mean/std metrics across folds
- [ ] **Decision**: If time/budget permits, execute during Day 5

---

## Issue #20: Final Documentation (Async Work)

**Objective**: Complete documentation with visual comparisons  
**Status**: 80% complete (README, Quick Start, Analysis done)  
**Estimated Time**: 1-2 hours (IDE/CPU work, no GPU)

### Task Checklist

#### 1. Visual Comparison Plots
- [ ] **Setup** (10 min)
  - [ ] Open new Jupyter notebook locally or on Colab CPU
  - [ ] Load best model checkpoint from Day 4 experiments
  - [ ] Load test set predictions (from analysis_results/)

- [ ] **Generate Overlays** (30 min)
  - [ ] Select 6 diverse test samples (mix of TP, FP, FN, TN)
  - [ ] For each sample, create 4-panel visualization:
    ```
    [Pre-Fire Image] | [Post-Fire Image]
    [Ground Truth]   | [Prediction @ 0.5] 
    ```
  - [ ] Add colormaps: Red = burn/prediction, Green = correct, Blue = miss
  - [ ] Save as: `docs/analysis_results/comparison_samples.png`

- [ ] **Threshold Visualization** (20 min)
  - [ ] Create grid showing predictions at thresholds: [0.3, 0.5, 0.7]
  - [ ] Show same 3 samples at each threshold
  - [ ] Demonstrate precision-recall tradeoff visually
  - [ ] Save as: `docs/analysis_results/threshold_comparison.png`

#### 2. Update Documentation
- [ ] **BASELINE_RESULTS.md**
  - [ ] Add section: "Day 4 Improvements" with new metrics
  - [ ] Link to DAY_4_SPRINT_PLAN.md
  - [ ] Update "Next Steps" based on Day 4 findings

- [ ] **Create DAY_4_RESULTS.md**
  - [ ] Summary of each issue (14, 15, 16)
  - [ ] Metrics table: baseline vs. improved model
  - [ ] Cost analysis: compute spent vs. improvement gained
  - [ ] Recommendations for future sprints
  - [ ] Save checkpoint locations

- [ ] **Update README.md**
  - [ ] Change "Day 3 (In Progress)" to "Day 3 ✅ Complete"
  - [ ] Add "Day 4: Model Optimization" with summary results
  - [ ] Update project status section

- [ ] **Checkpoint Documentation**
  - [ ] Create `docs/CHECKPOINT_GUIDE.md`
  - [ ] List all saved checkpoints with dates, metrics, and use cases
  - [ ] Show which checkpoint to load for:
    - [ ] Best precision (for regulatory reporting)
    - [ ] Best IoU (for balanced use)
    - [ ] Best recall (for comprehensive mapping)
    - [ ] Best F1 (for general purpose)

#### 3. Repository Cleanup
- [ ] **Git Commits**
  - [ ] Commit each day's results separately
  - [ ] Tag best model: `git tag best_model_v2.0`
  - [ ] Push to origin

- [ ] **Artifact Organization**
  - [ ] Move Day 4 results to: `docs/inference_runs/day4_improvements/`
  - [ ] Organize by issue: `day4_improvements/issue_14/`, `issue_15/`, etc.

### Issue #20 Success Metrics

| Deliverable | Status |
|-------------|--------|
| Visual comparison plots | TBD |
| Threshold comparison visualization | TBD |
| DAY_4_RESULTS.md created | TBD |
| README.md updated | TBD |
| CHECKPOINT_GUIDE.md created | TBD |
| All artifacts committed to git | TBD |

---

## Daily Schedule (Estimated)

### Morning (IDE Work) — 30 min
- [ ] Review BASELINE_ANALYSIS.md findings
- [ ] Prepare notebooks locally (copy, comment, organize)
- [ ] Create backup branch: `git checkout -b day4-improvements`

### Mid-Morning (GPU Training) — 1.5 hours
- [ ] **Issue #14, Task 1**: Class weighting (20 min train + 10 min eval)
  - Quick test to validate improvement direction
  - Decision: Proceed with class weights in all subsequent experiments

### Late Morning (GPU Training) — 1 hour
- [ ] **Issue #15, Task 2**: Loss ratio experiments (60 min train)
  - Run 3 different loss ratios in parallel if possible
  - Start with 20-epoch runs for quick feedback

### Lunch Break / Async Wait — GPU training running

### Afternoon (GPU Training) — 1 hour
- [ ] **Issue #14, Task 2**: Extended training (50 min train + 10 min eval)
  - Use best configuration from Issues #14 & #15
  - Train for 50 epochs, monitor early stopping

### Late Afternoon (CPU/Eval) — 1 hour
- [ ] Run full test set inference on all 3-4 final checkpoints
- [ ] Compute final metrics table
- [ ] Take notes for documentation

### Evening (IDE/Documentation) — 1-2 hours
- [ ] Create visual comparison plots
- [ ] Write DAY_4_RESULTS.md
- [ ] Update README and checkpoints guide
- [ ] Commit and tag results

**Total GPU Time**: ~3 hours (9% of monthly quota, total 15% with Day 3)

---

## Cost-Benefit Analysis

### ROI Calculation

| Experiment | GPU Cost | Expected Gain | ROI | Priority |
|------------|----------|---|---|---|
| Class weighting | 0.33 hrs | +7% recall, +1% IoU | HIGH | 1 |
| Extended training (50 ep) | 0.83 hrs | +3% IoU | MEDIUM | 2 |
| Loss tuning (3 seeds) | 1.0 hrs | +3% IoU, +2% F1 | MEDIUM | 3 |
| Learning rate search | 1.0 hrs | +1% IoU | LOW | 4 (defer) |
| **Total** | **3.16 hrs** | **+7-10% recall, +4-7% IoU** | **HIGH** | — |

### Conservative Path (Low Cost)
- [ ] Class weighting only (0.33 hrs)
- [ ] If successful, commit and document
- [ ] Defer extended training to Day 5

### Aggressive Path (Standard)
- [ ] Class weighting + Extended training + Loss tuning (3 hrs)
- [ ] Full evaluation and comparison
- [ ] High confidence in improvements

### Exploration Path (High Cost - Not Recommended)
- [ ] All experiments including learning rate search (4.2 hrs)
- [ ] Thorough ablation study
- [ ] Risk: May exceed 20% monthly quota

**Recommended**: Standard Path (3 hrs, ~10% total usage)

---

## Decision Gates

### Gate 1: After Class Weighting (Task #14.1)
```
IF recall improved ≥5%:
  THEN proceed to Extended Training and Loss Tuning
  ELSE try weight ratio 0.7/0.3 and re-evaluate
  OR document that class balance is not the limiting factor
```

### Gate 2: After Extended Training (Task #14.2)
```
IF IoU improved ≥2%:
  THEN keep 50-epoch configuration for final model
  ELSE recommend 20-30 epochs as optimum
  AND investigate other improvements (loss tuning, etc.)
```

### Gate 3: After Loss Tuning (Task #15.3)
```
IF IoU improved ≥2% over baseline:
  THEN adopt new loss ratio
  ELSE keep 50/50 as stable baseline
  AND note: class balance is primary lever (not loss ratio)
```

### Gate 4: Final Decision
```
IF total improvements (recall + IoU) ≥ expected (+7% recall, +3% IoU):
  THEN tag as "best_model_v2.0" and document improvements
  ELSE analyze failures and plan for Day 5 deep dive
```

---

## Contingency Plans

### If GPU Quota Runs Low
- [ ] Stop experiments and analyze results collected so far
- [ ] Defer Issue #16 (cross-validation) to future sprint
- [ ] Focus on documentation of findings
- [ ] Plan for efficiency improvements in Day 5

### If Class Weighting Doesn't Help
- [ ] Try focal loss instead (PyTorch built-in)
- [ ] Investigate data quality issues (potential label noise)
- [ ] Consider: model is already well-balanced by precision/recall tradeoff
- [ ] Pivot to Issue #15 (loss tuning as primary improvement lever)

### If No Improvement Observed
- [ ] Review baseline model: may already be near optimal for this dataset
- [ ] Investigate: Are hyperparameters limiting architecture, not tuning?
- [ ] Consider: Defer major improvements to Day 5+ (architecture changes)
- [ ] Recommend: Use current model in production, focus on deployment (#20)

---

## Success Criteria for Day 4

### Minimum Success
- [ ] Issue #14: Class weighting tested, clear result documented
- [ ] Issue #15: At least 2 loss ratios evaluated
- [ ] Issue #20: Visual comparison plots created
- [ ] Total GPU usage: ≤ 12% monthly quota

### Target Success
- [ ] Issue #14: Recall improved to 65%+ with class weighting
- [ ] Issue #15: Optimal loss ratio identified (+2-3% IoU)
- [ ] Issue #20: Full documentation with comparison visualizations
- [ ] Total GPU usage: 10-12% monthly quota
- [ ] All experiments committed to git with clear commit messages

### Stretch Success
- [ ] Issue #16: 3-fold cross-validation on best model
- [ ] Learning rate search completed (if time permits)
- [ ] Production deployment guide started
- [ ] Total GPU usage: ≤ 15% monthly quota

---

## Next Sprint Handoff (Day 5)

### Ready for Issue #20 Completion
- [ ] All Day 4 metrics and visualizations
- [ ] DAY_4_RESULTS.md with analysis
- [ ] Updated README and checkpoint guide

### Ready for Issue #16 (Cross-Validation)
- [ ] Best configuration identified
- [ ] Clear instructions for k-fold setup
- [ ] Estimated compute budget: 4-5 GPU hours

### Ready for Issue #23 (RGB-IR Transfer Learning)
- [ ] extract_rgbir.ipynb prepared
- [ ] Can run during Day 5 training downtime

---

## References

- Baseline Model: docs/BASELINE_RESULTS.md
- Test Set Analysis: docs/BASELINE_ANALYSIS.md
- Threshold Analysis: docs/analysis_results/threshold_analysis.png
- Recommendations: docs/BASELINE_ANALYSIS.md (Recommendations section)

---

**Author**: Stephen Cerruti  
**Created**: 2026-06-24  
**Status**: Ready for Day 4 Sprint Execution  
**Related Issues**: #13 (Day 4 Sprint), #14 (Hyperparameter Tuning), #15 (Loss Optimization), #16 (Cross-Validation), #20 (Final Documentation)
