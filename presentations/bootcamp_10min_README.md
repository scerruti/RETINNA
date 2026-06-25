# Bootcamp 10-Minute Presentation: Option C (Auctioneer Delivery)

## File
**Location**: `/Users/scerruti/RETINNA/presentations/bootcamp_10min_option_c.md`

**Format**: Slidev markdown (can be rendered with `slidev bootcamp_10min_option_c.md`)

## Presentation Structure (9 slides + Q&A)

| Slide | Title | Duration | Key Message |
|-------|-------|----------|-------------|
| 1 | Title Slide | ~15 sec | Hook: Data vs. model problem |
| 1b | Phase I Architecture | ~60 sec | U-Net, skip connections, 24 channels |
| 2 | Phase I Results & Discovery | ~60 sec | 0.52 IoU baseline + red flag |
| 3 | The Insight: Why It Failed | ~90 sec | **Label-data mismatch** (central pivot) |
| 4 | Phase II_01: Analytical Relabeling | ~75 sec | RdNBR + USGS thresholds |
| 5 | Phase II_01 Output | ~50 sec | 7-class distribution validation |
| 6 | Phase II_02: Architecture Redesign | ~70 sec | 8-channel RGBN for NAIP transfer |
| 7 | Phase II_02 Training Strategy | ~60 sec | Loss weights, normalization, scheduler |
| 8 | Results & PA3 Learning Objectives | ~75 sec | Training curves + 6 objectives ✓ |
| 9 | The Lesson & Next Steps | ~90 sec | **Data is the bottleneck** |
| 10 | Q&A | ~30 sec | Open discussion |

**Total**: ~11 minutes (includes natural pacing)

## Design Principles (Option C: Auctioneer Delivery)

✓ **Fast, high-impact visuals** - Minimal text on slides (5-10 words per slide)  
✓ **Speaker notes contain full narrative** - Not on slides  
✓ **Real data** - Embedded training curves from Phase I and Phase II_02  
✓ **Clean architecture diagrams** - ASCII/text format for clarity  
✓ **Clear narrative arc** - Problem → Investigation → Solution → Validation → Next Steps  

## Key Visuals

1. **Slide 2**: `loss_curves.png` - Phase I baseline training curves
2. **Slide 8**: `training_history_with_plateau_scheduler.png` - Phase II_02 results with scheduler validation

**Image paths in markdown**:
```
/Users/scerruti/RETINNA/docs/training_runs/baseline_24ch_epoch20/loss_curves.png
/Users/scerruti/RETINNA/docs/phase4_rgb_ir_training/training_history_with_plateau_scheduler.png
```

## Speaker Notes Summary

Every slide has detailed speaker notes (~1 min each) covering:
- **Slide 1**: Hook about data vs. model problem
- **Slide 1b**: U-Net architecture rationale (skip connections)
- **Slide 2**: Phase I results + the red flag moment
- **Slide 3**: Label-data mismatch investigation (PIVOT)
- **Slide 4**: RdNBR spectral relabeling approach
- **Slide 5**: Distribution validation (realistic for CA)
- **Slide 6**: 8-channel redesign for NAIP transfer
- **Slide 7**: Training strategy (loss, normalization, augmentation, scheduler)
- **Slide 8**: Results synthesis + PA3 objectives
- **Slide 9**: Key lesson + Phase III roadmap
- **Slide 10**: Q&A

## Rendering the Presentation

```bash
# If you have Slidev installed
slidev /Users/scerruti/RETINNA/presentations/bootcamp_10min_option_c.md

# Or use online viewer
# (copy markdown to slidev.dev)
```

## Content References

All information sourced from:
- `/Users/scerruti/RETINNA/docs/BASELINE_RESULTS.md` (Phase I metrics)
- `/Users/scerruti/RETINNA/docs/PHASE_II_MASTER.md` (Phase II overview)
- `/Users/scerruti/RETINNA/docs/ARCHITECTURE_RATIONALE.md` (U-Net design)
- `/Users/scerruti/RETINNA/docs/phase4_rgb_ir_training/TRAINING_COMPARISON_ANALYSIS.md` (Scheduler validation)
- Training image assets from `docs/training_runs/` and `docs/phase4_rgb_ir_training/`

## Delivery Tips

- **Pacing**: Let visuals breathe. Don't rush through numbers—let audience absorb.
- **Pivot point** (Slide 3): Slow down here. This is the insight.
- **Validation** (Slide 5): Emphasize QA checks. Shows rigor.
- **Impact statement** (Slide 9): "Data is the bottleneck, not the model" — pause before and after.
- **Q&A**: Be ready to discuss:
  - Why 8-channel vs difference images?
  - How RdNBR differs from dNBR?
  - What's the NAIP transfer strategy?
  - How did you identify the label-data mismatch?

---

**Status**: Ready for presentation  
**Created**: 2026-06-25  
**Duration**: 10 minutes (goal) + Q&A  
**Target**: PA3 bootcamp evaluation panel  
