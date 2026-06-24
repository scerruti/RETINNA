# Day 4 GPU & Compute Tracking

**Date**: 2026-06-24 (Day 4 Sprint)  
**Issue**: #14 (Hyperparameter Tuning - Class Weighting Experiment)

---

## Pre-Training Baseline (12:46 PM)

**Colab Pro Status:**
- Available compute units: **93.09** ⭐
- Usage rate: **1.27 per hour**
- Active sessions: 1
- Backend: Python 3 Google Compute Engine (GPU)

**Resources:**
```
System RAM:   1.5 / 51.0 GB   (3% used)
GPU RAM:      0.0 / 15.0 GB   (0% used) ← Clean slate
Disk:        47.1 / 235.7 GB  (20% used)
```

**Expected Training Cost:**
- Experiment: Class weighting (pos_weight=1.5) for 20 epochs
- Estimated GPU time: ~20 minutes = 0.33 GPU hours
- Expected compute units: ~0.33-0.50 units (rough estimate)
- Remaining after training: ~92.6-92.8 units

---

## Training Log

**Experiment 1: Class Weighting (pos_weight=1.5)**

| Metric | Value | Status |
|--------|-------|--------|
| Start Time | — | TBD |
| End Time | — | TBD |
| Epochs Completed | — | TBD |
| Best Val IoU | — | TBD |
| Best Epoch | — | TBD |
| Final Recall | — | TBD |
| Final Precision | — | TBD |
| Peak GPU RAM | — | TBD |
| Peak System RAM | — | TBD |
| Total Compute Units Used | — | TBD |

**Key Observations:**
- [ ] Did recall improve from 60.5%?
- [ ] By how much? (target: 65-70%)
- [ ] Did precision stay above 90%?
- [ ] GPU memory peak vs available (15GB)?

---

## Post-Training Status

**After Experiment 1 (est. 1:15 PM):**

Compute units remaining: — TBD —

Next experiment (if budget permits):
- [ ] Extended training (50 epochs)?
- [ ] Loss ratio tuning (30/70 split)?
- [ ] Or stop here and analyze results?

---

## Budget Tracking Summary

| Phase | GPU Hours | Compute Units | Cumulative |
|-------|-----------|---|---|
| Day 3 Baseline | 0.58 | ~2.5-3.0 | 3.0 |
| Day 4 Class Weighting | 0.33 | ~1.5-2.0 | 4.5-5.0 |
| Day 4 Extended Training (optional) | 0.83 | ~3.5-4.0 | 8.0-9.0 |
| Day 4 Loss Tuning (optional) | 1.0 | ~4.5-5.0 | 12.5-14.0 |
| **Recommended Total** | **~2 hrs** | **~10-12** | **~12-15** |

**Monthly Budget Analysis:**
- Total units available: ~1500 (Colab Pro)
- Current usage: 3% (baseline from Day 3)
- Day 4 planned: ~1% additional
- Remaining for rest of month: ~96%
- Safety buffer: Comfortable, no budget concerns ✓

---

## Notes

- Baseline established at clean state (0 GPU RAM)
- First priority: Confirm pos_weight=1.5 improves recall
- Secondary priority: If time/budget, test extended training
- Defer: Loss ratio tuning and learning rate search (can do Day 5)

**Ready to train!** ✓
