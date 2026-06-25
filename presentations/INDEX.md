# RETINNA Presentations Index

## Active Presentations

### 1. Bootcamp 10-Minute (Option C: Auctioneer Delivery) ✅ READY
**File**: `bootcamp_10min_option_c.md`  
**Status**: Complete and ready for delivery  
**Duration**: 10 minutes + Q&A  
**Format**: Slidev markdown  
**Audience**: PA3 bootcamp evaluation panel  
**Key Message**: "Data is the bottleneck, not the model"

**Structure**:
- 10 slides + Q&A
- Embedded training curve images (real data)
- Full speaker notes (~670 sec of narrative)
- Option C design: minimal slide text, visual-driven, high-impact

**Quick Start**:
```bash
slidev bootcamp_10min_option_c.md
```

**Key Slides**:
1. Title slide with hook
1b. Phase I U-Net architecture (24-channel)
2. Phase I results & red flag (0.52 IoU)
3. **PIVOT**: Label-data mismatch discovery
4. RdNBR-based spectral relabeling
5. 7-class distribution validation
6. Architecture redesign (8-channel NAIP-ready)
7. Training strategy (loss, norm, augmentation, scheduler)
8. Results & PA3 objectives
9. The lesson: "Data is the bottleneck"
10. Q&A

**Supporting Materials**:
- `bootcamp_10min_README.md` — Detailed slide guide
- `BOOTCAMP_CREATION_SUMMARY.md` — Complete creation notes

---

## Archive (Previous Versions)

- `10-min-PA3-pitch.md` — Earlier version (2026-06-24)
- `extended-PA3-talk.md` — Extended format (2026-06-24)

---

## Visualizations & Assets

**Training Curves** (embedded in presentation):
- `/Users/scerruti/RETINNA/docs/training_runs/baseline_24ch_epoch20/loss_curves.png` — Phase I baseline
- `/Users/scerruti/RETINNA/docs/phase4_rgb_ir_training/training_history_with_plateau_scheduler.png` — Phase II_02 with scheduler

**Source Documentation**:
- `/Users/scerruti/RETINNA/docs/BASELINE_RESULTS.md` — Phase I metrics
- `/Users/scerruti/RETINNA/docs/PHASE_II_MASTER.md` — Phase II overview
- `/Users/scerruti/RETINNA/docs/ARCHITECTURE_RATIONALE.md` — U-Net architecture rationale

---

## Presentation Checklist

### Before Delivery
- [ ] Verify Slidev installation: `slidev --version`
- [ ] Test image loading (both PNG files)
- [ ] Read speaker notes for each slide
- [ ] Practice timing (~11 min including transitions)
- [ ] Practice pivot point (Slide 3) delivery

### During Delivery
- [ ] Speak slowly and deliberately
- [ ] Let charts breathe (don't over-explain)
- [ ] Emphasize the insight (Slide 3: label mismatch)
- [ ] Pause before key message (Slide 9)
- [ ] Engage Q&A (NAIP transfer, RdNBR, methodology)

### Key Numbers to Remember
| Metric | Value |
|--------|-------|
| Phase I IoU | 0.52 |
| Phase I Parameters | 31.1M |
| Phase II Samples | 424 pairs → 848 images |
| Phase II Classes | 7 (RdNBR-based) |
| Scheduler Improvement | -6.2% val loss, +2.0% acc |

---

## Speaker Notes Summary

**Slide 1** (~15 sec): Hook — "wanted to detect burn scars..."
**Slide 1b** (~60 sec): U-Net architecture, skip connections, 24-channel input
**Slide 2** (~60 sec): Phase I results, IoU 0.52, the red flag moment
**Slide 3** (~90 sec): **PIVOT** — label-data mismatch discovery
**Slide 4** (~75 sec): RdNBR formula, USGS thresholds, spectral approach
**Slide 5** (~50 sec): 7-class distribution, validation checks
**Slide 6** (~70 sec): Architecture redesign rationale, NAIP transfer
**Slide 7** (~60 sec): Training strategy details (loss, norm, augmentation, scheduler)
**Slide 8** (~75 sec): Results, PA3 objectives achieved
**Slide 9** (~90 sec): **IMPACT** — "Data is the bottleneck", Phase III roadmap
**Slide 10** (~30 sec): Q&A

**Total**: ~11 minutes of content

---

## Narrative Arc

```
Problem (Slides 1-2)
    ↓
Red Flag (Slide 2)
    ↓
Investigation → INSIGHT (Slide 3)
    ↓
Solution → Implementation (Slides 4-7)
    ↓
Validation → Success (Slides 5, 8)
    ↓
Synthesis → Lesson (Slide 9)
    ↓
Next Steps & Q&A (Slides 9-10)
```

---

## Technical Coverage

✓ U-Net architecture (skip connections, encoder-decoder)  
✓ Spectral indices (NBR, RdNBR, MNDWI)  
✓ Loss functions (BCE+Dice, weighted cross-entropy)  
✓ Normalization (z-score, per-channel)  
✓ Data augmentation (flip, rotate, zoom/crop)  
✓ Learning rate scheduling (ReduceLROnPlateau)  
✓ Validation methodology (fold-based cross-validation)  
✓ Transfer learning (NAIP transition strategy)  

---

## Delivery Tips

**Pacing**: Let the visuals guide the pacing. Don't rush.

**Tone**: Confident, data-driven, problem-solving focused. Not apologetic about the pivot—it shows rigor.

**Emphasis**: 
- Slide 3 (insight): This is where the story turns. Slow down.
- Slide 5 (validation): "This distribution is realistic for California chaparral." Let that sink in.
- Slide 9 (lesson): Pause before and after the key message.

**Q&A Preparation**:
- "Why 8-channel instead of precomputed difference?" → Flexible learning, z-score benefits, augmentation
- "How does RdNBR differ from dNBR?" → Normalization by pre-fire vegetation, better for sparse landscapes
- "What's the NAIP transfer strategy?" → Model trained on NAIP-compatible bands (RGBN), zero-shot transfer
- "How did you identify the mismatch?" → Hyperparameter tuning made things worse, led to data investigation

---

## File Organization

```
presentations/
├── bootcamp_10min_option_c.md          [MAIN FILE - READY]
├── bootcamp_10min_README.md            [QUICK GUIDE]
├── BOOTCAMP_CREATION_SUMMARY.md        [DETAILED NOTES]
├── INDEX.md                            [THIS FILE]
├── 10-min-PA3-pitch.md                 [ARCHIVE]
├── extended-PA3-talk.md                [ARCHIVE]
└── visualizations/                     [SUPPLEMENTARY GRAPHICS]
```

---

## Status & Timeline

**Created**: 2026-06-25  
**Status**: ✅ READY FOR DELIVERY  
**Last Review**: 2026-06-25  
**Duration**: 10 minutes + Q&A  
**Word Count**: ~8,000 (slides + speaker notes)  

**Next Steps**:
1. Practice delivery (night before)
2. Verify image loading
3. Time the presentation
4. Have backup PDF copy
5. Deliver with confidence

---

**Questions?** See speaker notes in main presentation file.

