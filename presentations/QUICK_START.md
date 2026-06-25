# Quick Start Guide - PA3 Presentations

## Choose Your Presentation

### 10-Minute Version
**Best for**: Workshop intros, technical pitches, quick overviews  
**File**: `10-min-PA3-pitch.md`  
**Slides**: 6  
**Duration**: 10 minutes  
**Content**: Architecture, loss functions, validation results, learning objectives

**Start here if you have <15 minutes for audience questions and transitions.**

### Extended Version
**Best for**: In-depth lectures, graduate seminars, technical deep dives  
**File**: `extended-PA3-talk.md`  
**Slides**: 12  
**Duration**: 25-30 minutes  
**Content**: Full context, architecture comparisons, hyperparameter tuning, empirical analysis

**Start here if you want to explain the full journey from PA3 to wildfire detection.**

---

## How to View

### Option 1: Online with Slidev (Recommended)
Install Slidev first:
```bash
npm install -g @slidev/cli
```

Then launch:
```bash
# 10-minute version
slidev presentations/10-min-PA3-pitch.md

# Extended version
slidev presentations/extended-PA3-talk.md
```

Controls:
- Arrow keys: Navigate slides
- 'f': Fullscreen
- 'o': Overview (see all slides)
- 'p': Presenter notes
- 'b': Black screen (pause)

### Option 2: PDF Viewer (After Export)
```bash
# Export (requires Slidev installed)
slidev export presentations/10-min-PA3-pitch.md

# Open PDF
open presentations/10-min-PA3-pitch.pdf  # macOS
xdg-open presentations/10-min-PA3-pitch.pdf  # Linux
```

### Option 3: Text Editor (View as Markdown)
```bash
# Any text editor
cat presentations/10-min-PA3-pitch.md
# or open in VS Code, vim, etc.
```

Slide breaks are marked with `---`

---

## Key Content by Slide

### 10-Minute Presentation

| Slide | Topic | Duration | Key Takeaway |
|-------|-------|----------|--------------|
| 1 | Context & Motivation | 1 min | PA3 → RETINNA connection |
| 2 | FCN Architecture | 1.5 min | Encoder/decoder symmetry |
| 3 | U-Net vs FCN | 1.5 min | Skip connections matter |
| 4 | Loss Functions | 1.5 min | BCE + Dice for imbalance |
| 5 | Validation Results | 2 min | See false negatives visually |
| 6 | Learning Objectives | 2.5 min | 6 competencies achieved |

### Extended Presentation

| Slides | Topic | Duration | Key Takeaway |
|--------|-------|----------|--------------|
| 1 | Title & Context | 1 min | PA3 assignment foundation |
| 2 | Problem Domain | 2 min | Why wildfire detection matters |
| 3 | Data Overview | 2 min | 11-band Sentinel-2 data |
| 4 | Data Pipeline | 2 min | One-hot encoding + normalization |
| 5 | FCN Deep Dive | 3 min | Full architecture breakdown |
| 6 | U-Net Architecture | 2 min | Skip connections explained |
| 7 | Loss Functions | 2 min | BCE + Dice theory |
| 8 | Training Results | 2 min | Challenges identified |
| 9 | Hyperparameter Tuning | 3 min | Systematic optimization |
| 10 | FCN vs U-Net | 2 min | Empirical comparison |
| 11 | Validation Viz | 3 min | Pixel-level analysis |
| 12 | Learning Objectives | 3 min | 7 competencies achieved |

---

## Important Visualizations

### Pixel-Level False Negative Heatmap
**Shows**: Where the model makes mistakes  
**Color coding**:
- 🔴 Red = False Negatives (critical - missed burns)
- 🟢 Green = True Positives (correct detection)
- 🟡 Yellow = False Positives (acceptable error)
- ⚫ Gray = True Negatives (background)

**Key insight**: Model detects bright burns but misses lightweight slope fires

**Located in**: Slide 5 (10-min), Slide 11 (extended)

### Metrics Tables
**Shows**: Quantitative performance summary  
**Key metrics**:
- **IoU**: How much prediction overlaps ground truth
- **Precision**: When model says "burn", is it right?
- **Recall**: Does model find ALL burns? (Most important!)
- **FN Rate**: What percentage of burns are missed?

**Values from visualization**:
- IoU: 0.88 (strong overlap)
- Precision: 1.00 (high confidence)
- Recall: 0.88 (good coverage)
- FN Rate: 0.12 (12% missed)

---

## Presenter Tips

### For 10-Minute Version

**Before presenting**:
- Practice with timer - aim for exactly 10 minutes
- Highlight the color legend on Slide 5 heatmap
- Have backup explanation for why U-Net helps

**During presentation**:
- Emphasize learning objectives in final slide
- Take 30 seconds on false negative heatmap
- Mention that recall is most critical for wildfire
- Explain pos_weight=1.5 as "penalizing missed burns 1.5× more"

**Anticipate questions**:
- "Why not just use binary cross-entropy?" → Class imbalance (90/10 split)
- "What's a skip connection?" → Slide 3 shows this
- "How long to train?" → Covered in extended presentation

### For Extended Presentation

**Before presenting**:
- Read the full README.md for context
- Practice timing - aim for 25-30 minutes
- Have printed references for Sentinel-2 bands
- Prepare data visualization examples if possible

**During presentation**:
- Spend 3 minutes on Slide 4 (data pipeline) - it's complex
- Use Slide 8 training curves to show convergence
- Emphasize the hyperparameter results (Slide 9) as "what we learned"
- Slow down on Slide 11 heatmap - most impactful content
- End strong with Slide 12 learning objectives

**Engage the audience**:
- Ask: "What would you predict for lightweight burns?" before Slide 5
- Ask: "Why might this architecture struggle?" before explaining
- Pause on comparison charts (Slide 10) for questions
- Use failure modes (Slide 11) to discuss domain expertise needs

---

## File Locations

All files are in: `/Users/scerruti/RETINNA/presentations/`

**Presentations**:
- `10-min-PA3-pitch.md` ← View this
- `extended-PA3-talk.md` ← Or this

**Support files**:
- `visualizations/` → All PNG images
- `generate_visualizations.py` → Script to regenerate visuals
- `README.md` → Full documentation
- `10-min-PA3-pitch-plan.md` → Planning notes
- `extended-PA3-talk-plan.md` → Planning notes

---

## Customization

### Change Title or Author
Edit the YAML front matter in `.md` files:
```yaml
---
title: Your Custom Title
author: Your Name
date: 2026-06-24
---
```

### Add Speaker Notes
Between slides, add comments that won't display:
```markdown
<!-- 
Speaker note: Emphasize that recall is critical for wildfire detection
because missing a burn is worse than a false alarm.
-->
```

### Adjust Timing
For 5-minute version:
- Remove Slides 3 & 6 from 10-minute
- Keep architecture, loss function, results

For 45-minute version:
- Combine both presentations
- Add code snippets from actual implementation
- Include live demo of model predictions

---

## Export to PDF

### Using Slidev (Best Quality)
```bash
# Install Slidev globally (one-time)
npm install -g @slidev/cli

# Export presentation
slidev export presentations/10-min-PA3-pitch.md --out presentations/10-min-PA3-pitch.pdf

# Open
open presentations/10-min-PA3-pitch.pdf  # macOS
```

### Using Pandoc (Alternative)
```bash
brew install pandoc  # Install if needed

# Export (simple, may lose formatting)
pandoc presentations/10-min-PA3-pitch.md -o presentations/10-min-PA3-pitch.pdf
```

---

## Troubleshooting

### Visualizations not showing
- Check that `visualizations/` directory exists with PNG files
- Verify file paths in markdown use `./visualizations/filename.png`
- In Slidev, use `![Alt text](./visualizations/image.png)`

### Slidev won't launch
- Install: `npm install -g @slidev/cli`
- Update: `npm update -g @slidev/cli`
- Check Node version: `node --version` (should be 14+)

### Text too small/large
- Adjust in YAML front matter:
```yaml
---
fonts:
  sans: 'Roboto'
---
```
- Or use Slidev's zoom feature: `+/-` keys

### Can't find files
```bash
# Navigate to presentations directory first
cd /Users/scerruti/RETINNA/presentations
slidev 10-min-PA3-pitch.md
```

---

## Share Your Presentation

### Email
- Attach PDF export
- Attach README.md for context

### GitHub
- Push to your RETINNA repo
- Link to presentations/README.md

### Web
- Use Slidev's built-in server: `slidev --remote`
- Share the URL with viewers
- They can follow along in real-time

### Print
- Export to PDF
- Print with "fit to page" option
- Recommended: 4 slides per page

---

## Questions?

Refer to:
- **README.md** — Full technical documentation
- **10-min-PA3-pitch-plan.md** — Planning for short version
- **extended-PA3-talk-plan.md** — Planning for long version
- **COMPLETION_SUMMARY.md** — Project overview

---

**Status**: ✅ Ready to present!  
**Last Updated**: 2026-06-24  
**Slidev Version**: 0.48+ (check with `slidev --version`)
