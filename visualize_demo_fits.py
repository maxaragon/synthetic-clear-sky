#!/usr/bin/env python3
"""
Visualize the fitting strategy used for each demo.
Shows which pixels were used for fitting (single-band vs multi-band).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import sys

sys.path.insert(0, 'utils')
from colorspace_utils import srgb_to_linear

sys.path.insert(0, 'src')
from auto_optimize_clearsky import (
    build_geometry, fit_disk, load_semantic_mask, compute_sun_pixel_angle,
    ZE_HORIZON_CUTOFF, BAND_GAMMA_MIN, BAND_GAMMA_MAX
)

def create_multi_band_mask(SPA_deg, ze_deg, fitting_mask_base):
    """Create multi-band mask (4 bands)."""
    band1 = fitting_mask_base & (SPA_deg >= 88.0) & (SPA_deg <= 92.0)
    band2 = fitting_mask_base & (SPA_deg >= 80.0) & (SPA_deg < 88.0)
    band3 = fitting_mask_base & (SPA_deg >= 45.0) & (SPA_deg <= 75.0)
    band4 = fitting_mask_base & (ze_deg < 30.0)
    
    multi_band = band1 | band2 | band3 | band4
    
    stats = {
        'band1': band1.sum(),
        'band2': band2.sum(),
        'band3': band3.sum(),
        'band4': band4.sum(),
        'total': multi_band.sum()
    }
    
    return multi_band, stats

# Process all 5 demos
fig, axes = plt.subplots(2, 5, figsize=(25, 10))
fig.suptitle('Fitting Strategy Used for Each Demo\n(Single-Band vs Multi-Band)', 
             fontsize=20, fontweight='bold', y=0.98)

# Map demo numbers to their file extensions
demo_extensions = {
    1: 'png',
    2: 'webp',
    3: 'webp',
    4: 'jpg',
    5: 'png'
}

for demo_num in range(1, 6):
    print(f"\n=== Processing Demo {demo_num} ===")
    
    # Load image
    ext = demo_extensions[demo_num]
    img_path = f'demo/inputs/demo{demo_num}_image.{ext}'
    img = Image.open(img_path).convert('RGB')
    rgb = np.array(img, np.float32) / 255.0
    H, W = rgb.shape[:2]
    
    # Load masks
    class_masks = load_semantic_mask(f'demo/inputs/demo{demo_num}_mask.png')
    clear_sky_mask = class_masks.get('clear_sky', np.zeros((H, W), dtype=bool))
    sun_mask = class_masks.get('sun', np.zeros((H, W), dtype=bool))
    
    # Build geometry
    sky_combined = clear_sky_mask | sun_mask
    cx, cy, R = fit_disk(sky_combined.astype(np.uint8) * 255)
    theta, ze_deg, az_nav, disk = build_geometry(H, W, cx, cy, R, 'equisolid', K=1.4)
    
    # Sun position
    if sun_mask.any():
        ys_sun, xs_sun = np.nonzero(sun_mask)
        sun_x, sun_y = xs_sun.mean(), ys_sun.mean()
    else:
        sun_x, sun_y = cx, cy
    
    sun_y_int, sun_x_int = int(sun_y), int(sun_x)
    sun_az_nav = az_nav[sun_y_int, sun_x_int]
    sun_ze_deg = ze_deg[sun_y_int, sun_x_int]
    SPA_deg = compute_sun_pixel_angle(az_nav, ze_deg, sun_az_nav, sun_ze_deg)
    
    # Create masks
    non_horizon = (ze_deg <= ZE_HORIZON_CUTOFF)
    sun_exclusion = (SPA_deg <= 10.0) & disk
    fitting_mask_sky = clear_sky_mask & disk & non_horizon & (~sun_exclusion)
    
    single_band = fitting_mask_sky & (SPA_deg >= BAND_GAMMA_MIN) & (SPA_deg <= BAND_GAMMA_MAX)
    multi_band, band_stats = create_multi_band_mask(SPA_deg, ze_deg, fitting_mask_sky)
    
    # Load results to get actual band strategy used
    results_path = f'demo/outputs_sunmodel_v3/demo{demo_num}/optimization_results.json'
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    band_strategy_full = results.get('fitting_strategy', {}).get('band_strategy', 'unknown')
    band_strategy = 'single' if 'single' in band_strategy_full else 'multi'
    
    # Determine which mask was actually used
    if band_strategy == 'single':
        used_mask = single_band if single_band.sum() >= 50 else fitting_mask_sky
        strategy_name = 'Single-Band (88-92°)' if single_band.sum() >= 50 else 'Single-Band (Full)'
    else:
        used_mask = multi_band if multi_band.sum() >= 50 else fitting_mask_sky
        strategy_name = 'Multi-Band (4 zones)'
    
    # Create visualization overlay
    overlay = np.zeros((H, W, 3), dtype=np.float32)
    
    # Single-band visualization (top row)
    single_viz = rgb.copy()
    if single_band.any():
        # Highlight single-band region in bright green with better visibility
        single_viz[single_band] = single_viz[single_band] * 0.3 + np.array([0.0, 1.0, 0.0]) * 0.7
    
    # Multi-band visualization (bottom row)
    multi_viz = rgb.copy()
    if band_strategy == 'multi':
        # Color-code the 4 bands with better visibility
        band1 = fitting_mask_sky & (SPA_deg >= 88.0) & (SPA_deg <= 92.0)
        band2 = fitting_mask_sky & (SPA_deg >= 80.0) & (SPA_deg < 88.0)
        band3 = fitting_mask_sky & (SPA_deg >= 45.0) & (SPA_deg <= 75.0)
        band4 = fitting_mask_sky & (ze_deg < 30.0)
        
        if band1.any():
            multi_viz[band1] = multi_viz[band1] * 0.2 + np.array([1.0, 0.0, 0.0]) * 0.8  # Bright Red
        if band2.any():
            multi_viz[band2] = multi_viz[band2] * 0.2 + np.array([1.0, 0.5, 0.0]) * 0.8  # Bright Orange
        if band3.any():
            multi_viz[band3] = multi_viz[band3] * 0.2 + np.array([1.0, 1.0, 0.0]) * 0.8  # Bright Yellow
        if band4.any():
            multi_viz[band4] = multi_viz[band4] * 0.2 + np.array([0.0, 1.0, 0.0]) * 0.8  # Bright Green
    else:
        # Show single-band in bright green
        if single_band.any():
            multi_viz[single_band] = multi_viz[single_band] * 0.3 + np.array([0.0, 1.0, 0.0]) * 0.7
    
    # Plot
    col = demo_num - 1
    
    # Top row: Single-band visualization
    axes[0, col].imshow(single_viz)
    axes[0, col].set_title(f'Demo {demo_num}\nSingle-Band\n{single_band.sum():,} px', 
                           fontsize=11, fontweight='bold')
    axes[0, col].axis('off')
    
    # Bottom row: Multi-band or used strategy
    axes[1, col].imshow(multi_viz)
    if band_strategy == 'multi':
        title = f'Multi-Band\n{band_stats["total"]:,} px\n✅ USED'
    else:
        title = f'Single-Band\n{single_band.sum():,} px\n✅ USED'
    axes[1, col].set_title(title, fontsize=11, fontweight='bold', color='green')
    axes[1, col].axis('off')
    
    print(f"  Strategy: {strategy_name}")
    print(f"  Single-band: {single_band.sum():,} px")
    print(f"  Multi-band: {band_stats['total']:,} px")

# Add legend
legend_text = """
Single-Band: Green = Antisolar (88-92°)
Multi-Band: Red=88-92° | Orange=80-88° | Yellow=45-75° | Green=Zenith<30°
"""
fig.text(0.5, 0.02, legend_text, ha='center', fontsize=12, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout(rect=[0, 0.04, 1, 0.96])
plt.savefig('demo/outputs_sunmodel_v3/fitting_strategies_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Saved: demo/outputs_sunmodel_v3/fitting_strategies_comparison.png")
plt.close()

# Create summary statistics
print("\n" + "="*80)
print("FITTING STRATEGY SUMMARY")
print("="*80)

summary_data = []
for demo_num in range(1, 6):
    results_path = f'demo/outputs_sunmodel_v3/demo{demo_num}/optimization_results.json'
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    band_strategy_full = results.get('fitting_strategy', {}).get('band_strategy', 'unknown')
    band_strategy = band_strategy_full
    combined_error = results.get('final_result', {}).get('final_error', 0.0)
    
    print(f"Demo {demo_num}: {band_strategy:>12s} | Error: {combined_error:.4f}")
    summary_data.append((demo_num, band_strategy, combined_error))

print("="*80)
print(f"\nSingle-band: {sum(1 for _, s, _ in summary_data if 'single' in s)}/5 demos")
print(f"Multi-band:  {sum(1 for _, s, _ in summary_data if 'multi' in s)}/5 demos")

