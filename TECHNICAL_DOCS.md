# Synthetic Clear-Sky Generation - Production Version

## 🎯 Overview

This repository contains the production-ready pipeline for generating synthetic clear-sky images from real sky images with clouds. The pipeline removes clouds and generates physically-based clear-sky renderings using the Chauvin et al. (2015) photometric model.

## 📊 **Production Version: V22 (4-Stage Hybrid Optimizer)**

**V22** is the final production winner, achieving **2.7% better performance** than the previous best (V17).

### Architecture

1. **Stage 1: Grid Search**
   - Tests 4 sun size variants with both single-band and multi-band strategies
   - Smart band selection: prefers single-band when color quality is similar
   - Projection: equisolid (optimal for fisheye cameras)

2. **Stage 2: Continuous Refinement**
   - Refines clear-sky parameters using L-BFGS-B optimization
   - Optimizes 21 parameters (7 per RGB channel)
   - Only improves if error decreases

3. **Stage 3: Sun Enhancement**
   - Adds realistic sun disk + bloom PSF
   - Optimizes 4 sun parameters (intensity, radius, bloom sigma, bloom amplitude)
   - Uses L-BFGS-B with physics-based initial guesses

4. **Stage 4: Sun Parameter Grid Refinement** ⭐ **NEW**
   - Exhaustive grid search of 252 sun configurations
   - Tests: radius [0.3-0.8]×, intensity [0.1-0.7], bloom [5-20px]
   - Finds global optimum that Stage 3's L-BFGS-B missed
   - **Key to 2.7% improvement over V17**

5. **Smart Stage Selection**
   - Automatically selects best result from Stages 2-4
   - Uses the stage with lowest combined error
   - Prevents degradation from later stages

### Performance

| Test | V17 (3-Stage) | V22 (4-Stage) | Improvement |
|------|---------------|---------------|-------------|
| Test 1 (Feb 2025) | 0.1321 | **0.1295** | ✅ +2.0% |
| Test 2 (Jul 2025) | 0.1872 | 0.1872 | Same |
| Test 3 (Sep 2025) | 0.3665 | 0.3665 | Same |
| Test 4 (Mobotix) | 0.3315 | **0.3068** | ✅ +7.5% |
| Test 5 (Sep 2025b) | N/A | **0.1902** | New |
| **AVERAGE (1-4)** | **0.2543** | **0.2475** | ✅ **+2.7%** |

### Key Findings

**Optimal Sun Parameters** (from Stage 4 grid search):
- **Sun radius: 0.3-0.5×** solar angular radius (was 1.0× in V17)
- **Sun intensity: 0.1-0.3** (was 0.5 in V17)
- **Bloom sigma: 7-20px** (varies by image, was 10px in V17)

**Why Stage 4 Helps:**
- V17's L-BFGS-B optimizer gets stuck at local minima
- Initial guess (radius=1.0×) + regularization keeps it there
- Real-world images have hazy/cloudy aureoles → smaller sharp core
- Stage 4's exhaustive search finds the true global optimum

## 🚀 Usage

### Basic Usage

```bash
python src/hybrid_optimize_sun_v22.py \
    --image path/to/image.png \
    --mask path/to/mask.png \
    --output output/test_name
```

### Requirements

- Python 3.8+
- NumPy, OpenCV, scipy, matplotlib, Pillow
- Conda environment (recommended): `activate_WUR`

### Input Requirements

**Image Format:**
- RGB image (PNG, JPG, WEBP)
- Fisheye or hemispherical projection
- Any resolution (tested on 224×224 to 512×512)

**Semantic Mask Format:**
- PNG image with labeled regions:
  - Class 0: Background (black)
  - Class 1: Clear sky (red)
  - Class 2: Sun (green)
  - Class 3: Cloud (blue)
  - Class 4: Horizon/obstacles (optional)

### Output

The script generates three files in the output directory:

1. `best_synthetic_clearsky.png`: Final synthetic clear-sky image
2. `best_comparison.png`: 2×2 comparison (real, synthetic, sun zoom, difference)
3. `optimization_results.json`: Detailed results including:
   - Errors for all 4 stages
   - Final parameters used
   - Stage selection decision
   - Sun parameters (radius, intensity, bloom)

## 📁 Repository Structure

```
synthetic-clear-sky/
├── src/
│   ├── hybrid_optimize_sun_v22.py    # Production version (V22)
│   ├── hybrid_optimize_sun_v17.py    # Previous best (V17) for comparison
│   ├── auto_optimize_clearsky.py     # Baseline brute-force grid search
│   ├── clearsky_model.py             # Chauvin model implementation
│   └── colorspace_utils.py           # sRGB ↔ linear RGB conversions
├── test_data/
│   ├── image1_feb2025.png            # Test image 1
│   ├── image1_feb2025_mask.png       # Semantic mask 1
│   ├── image2_jul2025.webp           # Test image 2
│   ├── image2_jul2025_mask.png       # Semantic mask 2
│   ├── image3_sep2025.webp           # Test image 3
│   ├── image3_sep2025_mask.png       # Semantic mask 3
│   ├── image4_mobotix.jpg            # Test image 4 (Mobotix camera)
│   ├── image4_mobotix_mask.png       # Semantic mask 4
│   ├── image5_sep2025b.png           # Test image 5
│   └── image5_sep2025b_mask.png      # Semantic mask 5
├── output/
│   ├── hybrid_v17_test*              # V17 results (for comparison)
│   └── hybrid_v22_test*              # V22 results (production)
└── PRODUCTION_README.md              # This file
```

## 🔬 Technical Details

### Photometric Model (Chauvin et al. 2015)

Clear-sky radiance model:
```
L(θ, γ) = G(θ) × S(γ)
```

Where:
- **G(θ)**: Gradation function (zenith angle dependence)
  - `G(θ) = A × cos(θ)^C + B`
  - Fitted independently for R, G, B channels
  
- **S(γ)**: Scattering function (sun-pixel angle dependence)
  - `S(γ) = D + E × γ^F + H × cos(γ)`
  - Fitted independently for R, G, B channels

### Sun Model

The sun is modeled separately in Stage 3/4:

**Sun Disk** (angular space):
- Smooth sigmoid transition at solar angular radius (0.266°)
- Scaled by radius factor (0.3-0.8× optimal)

**Bloom/PSF** (pixel space):
- Gaussian falloff: `amplitude × exp(-0.5 × (r/σ)²)`
- Represents optical scatter and sensor bloom
- Sigma: 7-20px (image-dependent)

### Projection

**Equisolid** projection (for fisheye cameras):
```
r = 2R × sin(θ/2)
```

Where:
- `r`: Radial distance from optical center
- `R`: Image radius (fitted from mask disk)
- `θ`: Zenith angle

### Error Metric

Combined error (used for stage selection):
```
Error = 0.7 × Clear-Sky-Error + 0.3 × Sun-Error
```

Both errors use L1 norm in linear RGB space.

## 🔄 Version History

### Major Versions

- **V1-V9**: Initial development (multi-band fitting, projection selection)
- **V10**: Smart band selection (single vs multi-band)
- **V17**: Color-aware band selection (**previous production winner**)
- **V22**: + Stage 4 Sun Grid Refinement (**current production winner, +2.7%**)
- **V23**: + Stage 5 Color Transfer (tested, rejected - degraded results)

### V17 vs V22 Comparison

| Feature | V17 (3-Stage) | V22 (4-Stage) |
|---------|---------------|---------------|
| Grid Search | ✓ | ✓ |
| Continuous Refinement | ✓ | ✓ |
| Sun Enhancement (L-BFGS-B) | ✓ | ✓ |
| Sun Grid Search (252 configs) | ✗ | ✓ ⭐ |
| Smart Stage Selection | 3 stages | 4 stages |
| Average Error | 0.2543 | **0.2475** |
| Improvement | Baseline | **+2.7%** |

## 🎓 References

- Chauvin, R., Nou, J., Thil, S., Traoré, A., Grieu, S. (2015). "Cloud detection methodology based on a sky-imaging system." *Energy Procedia*, 69, 1970-1980.

## 📧 Contact

For questions or issues, contact the PhD research team at WUR.

---

**Last Updated**: 2025-01-18  
**Production Version**: V22  
**Status**: ✅ Production Ready

