# Official Sentinel-2 Burn Classification Standards

**Purpose**: Reference authoritative sources for multi-class burn severity classification  
**Date**: 2026-06-24  
**Status**: External standards compiled

---

## Primary Authority: USGS MTBS (Monitoring Trends in Burn Severity)

**Official Website**: https://www.mtbs.gov/

**MTBS Standard Classification** (based on dNBR thresholds):

| Class | dNBR Range | Description |
|-------|-----------|-------------|
| Unburned | < 0.1 | No burn impact or minimal change |
| Low Severity | 0.1 to 0.27 | Light burn, vegetation survives |
| Moderate Severity | 0.27 to 0.44 | Substantial vegetation loss |
| High Severity | 0.44 to 0.66 | Severe burn, complete canopy kill |
| Extreme/Very High Severity | ≥ 0.66 | Complete consumption, terrain change |

**Source**: USGS Remote Sensing Phenology Lab, adopted by USDA Forest Service nationally

---

## Key Academic Foundation

### 1. Key et al. (2006)
**Title**: "The Normalized Burn Ratio (NBR): A Landsat measure for measuring burn severity"  
**Journal**: Remote Sensing of Environment, vol. 29  
**Key Finding**: 
- NBR = (NIR - SWIR) / (NIR + SWIR)
- dNBR = NBRpre - NBRpost captures burn intensity
- Thresholds validated against field observations

### 2. Miller & Thode (2007)
**Title**: "Quantifying burn severity in a heterogeneous landscape with a relative version of the delta Normalized Burn Ratio (dNBR)"  
**Journal**: Remote Sensing of Environment, vol. 109  
**Key Finding**:
- Relative dNBR = dNBR / (|NBRpre| + 0.005)
- Normalizes for pre-fire conditions
- Better for heterogeneous landscapes (different biomes)

---

## Sentinel-2 Specific Implementation

### ESA Sentinel-2 Technical Handbook
**Reference**: https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-2-msi

**Recommended Bands for Burn Detection**:
- **B8 (NIR, 842 nm)**: Vegetation absorption
- **B11 (SWIR-1, 1610 nm)**: Burn indicator (USGS standard)
- **B12 (SWIR-2, 2190 nm)**: Alternative SWIR
- **B4 (Red, 665 nm)**: Soil exposure
- **B3 (Green, 560 nm)**: Water detection
- **B2 (Blue, 490 nm)**: Cloud detection

**Scene Classification Layer (SCL)**: 
- Includes cloud/shadow classification (classes 0, 1, 2, 3, 8, 9, 10, 11)
- Should be used for masking invalid pixels

---

## Water Detection (Complementary Standard)

**Source**: McFEE et al. (2013) - "Water body mapping with Sentinel-2 data"

**Modified Normalized Difference Water Index (MNDWI)**:
```
MNDWI = (Green - SWIR) / (Green + SWIR)
Threshold: MNDWI > 0.3 → Water
```

---

## Critical Implementation Choices

### Band Selection for S2
Based on USGS MTBS applied to Sentinel-2:

**Required**:
- B8 (NIR) - essential for NBR
- B11 (SWIR-1) - USGS standard (NOT B12 or CLP)

**Recommended for Enhanced Accuracy**:
- B4 (Red) - NDVI, soil exposure detection
- B3 (Green) - MNDWI, water detection
- B2 (Blue) - cloud detection, spectral coherence

**Optional (Not Essential)**:
- B5, B6, B7 (Red Edge) - vegetation stress
- B8A (Narrow NIR) - spectral detail
- B12 (SWIR-2) - alternative SWIR (less common for burns)

### Threshold Validation
- **USGS MTBS thresholds**: Originally developed for Landsat (30m)
- **For Sentinel-2 (20m)**: Apply same thresholds (conservative approach)
- **Alternative**: Empirically validate on in-situ data if available

---

## Recommended Classification Algorithm

**Stage 1: Compute Core Indices**
```
NBR_pre = (B8_pre - B11_pre) / (B8_pre + B11_pre)
NBR_post = (B8_post - B11_post) / (B8_post + B11_post)
dNBR = NBR_pre - NBR_post

MNDWI = (B3 - B11) / (B3 + B11)  # post-fire
```

**Stage 2: Apply USGS Thresholds (Priority Order)**
```
IF SCL in [0,1,2,3,8,9,10,11]:
    Class = 6 (Cloud/Shadow)
ELIF MNDWI > 0.3:
    Class = 5 (Water)
ELIF dNBR < -0.27:
    Class = 4 (Extreme Severity)
ELIF -0.27 ≤ dNBR ≤ -0.1:
    Class = 3 (High Severity)
ELIF -0.1 < dNBR ≤ 0.05:
    Class = 2 (Moderate Severity)
ELIF 0.05 < dNBR ≤ 0.27:
    Class = 1 (Low Severity)
ELSE:
    Class = 0 (Unburned)
```

**Stage 3 (Optional - Enhanced Accuracy)**
- Compute NDVI change: ΔNDVI = NDVI_pre - NDVI_post
- Check coherence: Do dNBR and ΔNDVI agree?
- Flag low-coherence pixels as ambiguous
- Use B2/B3/B8 ratio for cloud confidence

---

## Expected Class Distribution

For typical California fire regimes (from MTBS national database):

| Class | Typical % | Range |
|-------|-----------|-------|
| Unburned | 40-55% | 30-70% |
| Low Severity | 5-15% | 2-20% |
| Moderate Severity | 15-30% | 10-40% |
| High Severity | 10-25% | 5-30% |
| Extreme Severity | 2-8% | 1-15% |
| Water | 1-5% | 0-10% |
| Cloud/Shadow | 2-5% | 1-10% |

---

## Quality Control & Validation

### Spatial Coherence Checks
- Burn patches should be spatially contiguous (not random scattered pixels)
- Fire boundaries should align with topography/vegetation patterns
- Large uniform areas of single class are suspicious

### Spectral Coherence Checks
- Check that multiple indices agree (dNBR, NDVI change, reflectance change)
- Pixels with contradictory signals → likely ambiguous/mixed

### Historical Validation
- Compare to USGS MTBS reference fires (same areas, Landsat-based)
- Expected agreement: ~80-90% for major burn classes

---

## Important Caveats

### Landsat vs Sentinel-2 Differences
- **Landsat**: 30m resolution, 16-day repeat
- **Sentinel-2**: 20m resolution (B8/B11), 10m (RGB), 5-day repeat
- USGS thresholds apply to both but S2 provides finer detail

### Pre/Post Image Timing
- **Ideal**: 1-2 weeks post-fire (peak spectral change)
- **Too soon**: May underestimate severity (still smoldering)
- **Too late**: Vegetation recovery obscures severity (1+ month)
- **Old scars**: Vegetation recovery makes severity unrecognizable

### Mixed Pixels & Heterogeneity
- At 20m resolution, fire boundaries are blended
- Partially burned pixels may not fit strict thresholds
- Relative dNBR (Miller & Thode 2007) handles this better

---

## Recommended Implementation Order

1. **Implement USGS MTBS basic algorithm** (5 min)
   - Core: dNBR thresholds on B8 + B11
   - Add: MNDWI for water

2. **Add enhanced indices** (10 min)
   - NDVI change
   - Coherence checking
   - Cloud masking via SCL or spectral

3. **Validate on known burns** (30 min)
   - Compare to MTBS reference data
   - Adjust thresholds if needed

4. **Quality control** (1 hour)
   - Visual inspection of predictions
   - Check spatial coherence
   - Sample verification

---

## References

1. USGS MTBS: https://www.mtbs.gov/
2. USGS Burn Severity FAQ: https://www.usgs.gov/faqs/what-normalized-burn-ratio-nbr
3. ESA Sentinel-2 Handbook: https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-2-msi
4. USDA Forest Service Burn Severity: https://www.fs.fed.us/
5. Key, C.H., Benson, N.C., 2006. Landscape Assessment: Ground measure of severity, the Composite Burn Index; and remote sensing of severity, the Normalized Burn Ratio. USDA Forest Service, Rocky Mountain Research Station. RMRS-GTR-164-CD.
6. Miller, J.D., Thode, A.E., 2007. Quantifying burn severity in a heterogeneous landscape with a relative version of the delta Normalized Burn Ratio (dNBR). Remote Sensing of Environment 109, 66-80.

---

**Authoritative Standard**: USGS MTBS (Monitoring Trends in Burn Severity)  
**For Sentinel-2**: Apply USGS thresholds with B8 (NIR) + B11 (SWIR-1)  
**Enhancement**: Add NDVI change, SCL cloud masking, spectral coherence checks

