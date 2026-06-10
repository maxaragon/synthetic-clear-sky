#!/usr/bin/env python3
"""
SYNTHETIC CLEAR-SKY OPTIMIZER - REALISTIC SUN MODEL V2

Key improvements for realistic sun appearance:
1. Extract actual sun color/intensity from real image (preserve saturation)
2. Use real sun pixels as-is in core region (no parametric model there)
3. Only model the aureole/glow transition to sky
4. Smooth blending using Gaussian weights
5. Preserve camera-specific optical characteristics

Strategy:
- Core (γ < 1°): Use real image pixels directly
- Transition (1° < γ < 8°): Blend real→model using distance-based weights
- Sky (γ > 8°): Pure Chauvin model

Author: Max Aragon, Mines Paris PSL
"""

from pathlib import Path
import os
import json
import numpy as np
from PIL import Image
import cv2
import sys
import argparse
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from colorspace_utils import srgb_to_linear, linear_to_srgb

from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter, binary_erosion
import matplotlib.pyplot as plt

# Import grid search functions
from auto_optimize_clearsky import (
    build_geometry, fit_disk, load_semantic_mask,
    compute_sun_pixel_angle, G_model, S_model, circular_mask,
    fit_per_channel, generate_synthetic as generate_synthetic_grid,
    evaluate_synthetic, ZE_HORIZON_CUTOFF
)

# Banding defaults for geometric fitting regions (narrower than legacy defaults).
# Can be overridden via environment vars for fast experiments.
BAND1_GAMMA_MIN = float(os.environ.get("BAND1_GAMMA_MIN", "89.0"))
BAND1_GAMMA_MAX = float(os.environ.get("BAND1_GAMMA_MAX", "91.0"))
BAND2_GAMMA_MIN = float(os.environ.get("BAND2_GAMMA_MIN", "82.0"))
BAND2_GAMMA_MAX = float(os.environ.get("BAND2_GAMMA_MAX", "89.0"))
BAND3_GAMMA_MIN = float(os.environ.get("BAND3_GAMMA_MIN", "50.0"))
BAND3_GAMMA_MAX = float(os.environ.get("BAND3_GAMMA_MAX", "72.0"))
BAND4_ZE_MAX = float(os.environ.get("BAND4_ZE_MAX", "28.0"))

# Single-band compatibility values (legacy channel): keep sun-antisolar neighborhood tight.
BAND_GAMMA_MIN = float(os.environ.get("BAND_GAMMA_MIN", "88.5"))
BAND_GAMMA_MAX = float(os.environ.get("BAND_GAMMA_MAX", "91.5"))

SAFE_SAMPLING_ZENITH_MAX_DEG = 75.0
SAFE_SAMPLING_EROSION_ITERS = 6
SAFE_SAMPLING_MIN_PIXELS = 3000
SUN_EXCLUSION_RADIUS_DEG = 10.0

# ==================== REALISTIC SUN BLENDING ====================

class RealisticSunBlender:
    """
    Blend real sun with synthetic sky using realistic transition.
    
    Strategy:
    1. Keep real sun pixels in core region (high fidelity)
    2. Model only the transition/aureole zone
    3. Smooth blending using Gaussian distance weights
    4. EXCLUDE cloud pixels from preservation (only sun + clear sky)
    5. INTERPOLATE missing sun pixels (clouds) using nearby clear sun values
    """
    
    def __init__(self, rgb_real_lin, gamma_deg, sun_mask, disk_mask, cloud_mask):
        self.rgb_real_lin = rgb_real_lin
        self.gamma_deg = gamma_deg
        self.sun_mask = sun_mask
        self.disk_mask = disk_mask
        self.cloud_mask = cloud_mask
        
        # Define regions
        self.CORE_RADIUS = 1.5      # Keep real image (high fidelity)
        self.TRANSITION_START = 1.5  # Start blending
        self.TRANSITION_END = 8.0    # End blending, pure sky
        
        # Core: use real pixels (EXCLUDE CLOUDS!)
        self.core_mask = (
            (gamma_deg <= self.CORE_RADIUS) & 
            disk_mask & 
            (~cloud_mask)  # CRITICAL: Don't preserve cloud pixels
        )
        
        # Transition: blend real→sky (EXCLUDE CLOUDS!)
        self.transition_mask = (
            (gamma_deg > self.TRANSITION_START) & 
            (gamma_deg <= self.TRANSITION_END) & 
            disk_mask &
            (~cloud_mask)  # CRITICAL: Don't blend cloud pixels
        )
        
        # Cloud gaps in sun region - will be interpolated
        self.cloud_gaps = cloud_mask & (gamma_deg <= self.TRANSITION_END) & disk_mask
        
        print(f"  Sun regions (clouds excluded):")
        print(f"    Core (real, γ<{self.CORE_RADIUS}°): {self.core_mask.sum():,} px")
        print(f"    Transition (blend, {self.TRANSITION_START}°<γ<{self.TRANSITION_END}°): {self.transition_mask.sum():,} px")
        
        # Cloud pixels in sun region - will be interpolated
        if self.cloud_gaps.any():
            print(f"    Cloud gaps (will interpolate): {self.cloud_gaps.sum():,} px")
    
    def interpolate_sun_gaps(self, rgb_sun):
        """
        Interpolate missing sun values in cloud gaps using radial basis functions.
        
        Strategy:
        - Use clear sun pixels (core + transition) as reference
        - Interpolate based on angular distance (gamma)
        - Apply per-channel RGB interpolation
        """
        if not self.cloud_gaps.any():
            return rgb_sun
        
        # Combined sun region (core + transition, no clouds)
        sun_region_clear = self.core_mask | self.transition_mask
        
        if not sun_region_clear.any():
            return rgb_sun  # No clear sun pixels to interpolate from
        
        # Get gamma values for interpolation
        gamma_clear = self.gamma_deg[sun_region_clear]
        gamma_gaps = self.gamma_deg[self.cloud_gaps]
        
        # Interpolate each channel
        rgb_interpolated = rgb_sun.copy()
        
        for ch in range(3):
            values_clear = rgb_sun[sun_region_clear, ch]
            
            # Radial basis function: weight by inverse squared distance in gamma space
            # For each gap pixel, compute weighted average of clear pixels
            for i, g_gap in enumerate(gamma_gaps):
                # Distance in gamma space
                dist = np.abs(gamma_clear - g_gap)
                
                # Weight: inverse squared distance + small epsilon
                weights = 1.0 / (dist**2 + 0.1)
                
                # Weighted average
                interpolated_value = np.sum(weights * values_clear) / np.sum(weights)
                
                # Assign to gap pixel
                gap_indices = np.where(self.cloud_gaps)
                rgb_interpolated[gap_indices[0][i], gap_indices[1][i], ch] = interpolated_value
        
        n_filled = self.cloud_gaps.sum()
        print(f"    ✅ Interpolated {n_filled:,} cloud gap pixels")
        
        return rgb_interpolated
    
    def create_blend_weights(self):
        """
        Create smooth blend weights: 1.0 (real) → 0.0 (synthetic sky).
        
        Uses smooth Gaussian-like falloff for natural appearance.
        """
        weights = np.zeros_like(self.gamma_deg, dtype=np.float32)
        
        # Core: 100% real
        weights[self.core_mask] = 1.0
        
        # Transition: smooth falloff
        in_transition = self.transition_mask
        if in_transition.any():
            gamma_trans = self.gamma_deg[in_transition]
            # Smooth cubic falloff
            normalized = (gamma_trans - self.TRANSITION_START) / (self.TRANSITION_END - self.TRANSITION_START)
            normalized = np.clip(normalized, 0, 1)
            # Use smoothstep for natural falloff
            smooth_weight = 1.0 - (3 * normalized**2 - 2 * normalized**3)
            weights[in_transition] = smooth_weight
        
        return weights
    
    def fit_transition_correction(self, sky_synthetic_lin):
        """
        Fit a simple correction to match transition zone intensity/color.
        
        This ensures smooth appearance where real sun meets synthetic sky.
        """
        # Use pixels in outer transition zone (closer to sky)
        outer_transition = (
            (self.gamma_deg > 4.0) & 
            (self.gamma_deg <= 7.0) & 
            self.disk_mask
        )
        
        if outer_transition.sum() < 100:
            print("    WARNING: Insufficient transition pixels, using defaults")
            return np.array([1.0, 1.0, 1.0])  # No correction
        
        # Compare real vs synthetic in this zone
        real_median = np.array([
            np.median(self.rgb_real_lin[..., i][outer_transition]) 
            for i in range(3)
        ])
        
        sky_median = np.array([
            np.median(sky_synthetic_lin[..., i][outer_transition]) 
            for i in range(3)
        ])
        
        # Correction factors
        correction = np.where(sky_median > 1e-6, real_median / sky_median, 1.0)
        correction = np.clip(correction, 0.5, 2.0)  # Reasonable bounds
        
        print(f"    Transition correction: R={correction[0]:.3f}, G={correction[1]:.3f}, B={correction[2]:.3f}")
        
        return correction
    
    def blend(self, sky_synthetic_lin):
        """
        Create final blended image: real sun + synthetic sky.
        
        Steps:
        1. Interpolate cloud gaps in sun region
        2. Create blend weights
        3. Apply transition correction
        4. Blend real sun + synthetic sky
        5. Smooth transition zone
        
        Returns:
            Blended linear RGB image, blend weights
        """
        # STEP 1: Interpolate cloud gaps (fill missing sun pixels)
        rgb_real_filled = self.interpolate_sun_gaps(self.rgb_real_lin)
        
        # STEP 2: Create blend weights
        blend_weights = self.create_blend_weights()
        
        # STEP 3: Apply transition correction to sky
        correction = self.fit_transition_correction(sky_synthetic_lin)
        sky_corrected = sky_synthetic_lin * correction
        
        # STEP 4: Blend using filled real image
        H, W = self.gamma_deg.shape
        blended = np.zeros((H, W, 3), dtype=np.float32)
        
        for i in range(3):
            blended[..., i] = (
                blend_weights * rgb_real_filled[..., i] +
                (1.0 - blend_weights) * sky_corrected[..., i]
            )
        
        # STEP 5: Apply gentle smoothing only in transition zone for seamless blend
        transition_smoothed = blended.copy()
        for i in range(3):
            channel_smooth = gaussian_filter(blended[..., i], sigma=1.5)
            blended[..., i] = np.where(
                self.transition_mask,
                channel_smooth,
                blended[..., i]
            )
        
        blended[~self.disk_mask] = 0
        
        return blended, blend_weights

# ==================== CONTINUOUS REFINEMENT ====================

class ContinuousRefiner:
    """Refines Chauvin parameters (excluding sun)."""
    
    def __init__(self, rgb_real, class_masks, theta, gamma, disk, ze_deg, fitting_method, sun_exclusion_mask,
                 sampling_mask=None):
        self.rgb_real = rgb_real
        self.rgb_lin = srgb_to_linear(rgb_real)
        self.theta = theta
        self.gamma = gamma
        self.disk = disk
        self.ze_deg = ze_deg
        self.fitting_method = fitting_method
        
        # Masks
        self.clear_sky_mask = class_masks.get("clear_sky", np.zeros_like(disk))
        self.sun_mask = class_masks.get("sun", np.zeros_like(disk))
        self.cloud_mask = class_masks.get("cloud", np.zeros_like(disk))
        
        # Exclude sun core from fitting (but include in evaluation!)
        non_horizon = (ze_deg <= ZE_HORIZON_CUTOFF)
        legacy_mask = self.clear_sky_mask & disk & non_horizon & (~sun_exclusion_mask)
        self.fitting_mask = sampling_mask if sampling_mask is not None else legacy_mask
        
        print(f"  Clear-sky fitting region: {self.fitting_mask.sum():,} pixels (safe sampling mask)")
        
        self.real_median = np.array([np.median(self.rgb_lin[..., i][self.fitting_mask]) 
                                     for i in range(3)])
    
    def generate_synthetic(self, params, E_scale=1.0, F_scale=1.0):
        """Generate synthetic from Chauvin parameters."""
        H_img, W_img = self.theta.shape
        rgb_syn_lin = np.zeros((H_img, W_img, 3), dtype=np.float32)
        
        if self.fitting_method == "per_channel":
            for i, ch_params in enumerate([params[0:7], params[7:14], params[14:21]]):
                A, B, C, D, E, F, H_coeff = ch_params
                E_scaled = E * E_scale
                F_scaled = F * F_scale
                
                G = G_model(self.theta, A, B, C)
                S = S_model(self.gamma, D, E_scaled, F_scaled, H_coeff)
                rgb_syn_lin[..., i] = G * S
        else:
            A, B, C, D, E, F, H = params[:7]
            scale_R, scale_G, scale_B = params[7:10]
            
            E_scaled = E * E_scale
            F_scaled = F * F_scale
            
            G = G_model(self.theta, A, B, C)
            S = S_model(self.gamma, D, E_scaled, F_scaled, H)
            Y_syn = G * S
            
            rgb_syn_lin[..., 0] = Y_syn * scale_R
            rgb_syn_lin[..., 1] = Y_syn * scale_G
            rgb_syn_lin[..., 2] = Y_syn * scale_B
        
        rgb_syn_lin = cv2.GaussianBlur(rgb_syn_lin, (5, 5), sigmaX=2.0)
        rgb_syn_lin[~self.disk] = 0
        
        return rgb_syn_lin
    
    def compute_loss(self, params, E_scale=1.0, F_scale=1.0):
        """Compute loss including sky AND sun regions (like original optimizer)."""
        rgb_syn_lin = self.generate_synthetic(params, E_scale, F_scale)
        
        syn_median = np.array([np.median(rgb_syn_lin[..., i][self.fitting_mask]) 
                               for i in range(3)])
        
        if np.any(syn_median < 1e-6):
            return 1e10
        
        scale_factors = self.real_median / syn_median
        rgb_syn_lin_scaled = rgb_syn_lin * scale_factors
        
        # Use full evaluation including sun (like original optimizer)
        # This forces sky colors to match the transition to sun
        clearsky_error, sun_error, combined_error = evaluate_synthetic(
            rgb_syn_lin_scaled, self.rgb_real, self.clear_sky_mask, self.sun_mask,
            self.disk, self.ze_deg, cloud_mask=self.cloud_mask
        )
        
        return combined_error if not np.isnan(combined_error) else 1e10
    
    def refine(self, initial_params, E_scale=1.0, F_scale=1.0, maxiter=100):
        """Refine parameters."""
        if self.fitting_method == "per_channel":
            bounds = [
                (0.0, None), (-2.0, 0.0), (-2.0, 1.5),
                (0.0, None), (0.0, 1e4), (0.2, 3.0), (-1.0, 1.0),
            ] * 3
        else:
            bounds = [
                (1e-3, None), (-0.9, 0.0), (-2.0, 2.0),
                (1e-3, None), (1e-3, 1e4), (0.2, 3.0), (-1.0, 1.0),
                (0.3, 3.0), (0.3, 3.0), (0.3, 3.0),
            ]
        
        result = minimize(
            lambda p: self.compute_loss(p, E_scale, F_scale),
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-9, 'maxiter': maxiter, 'disp': False}
        )
        
        return result

# ==================== MULTI-BAND FITTING ====================

def create_multi_band_mask(SPA_deg, ze_deg, fitting_mask):
    """Create multi-band mask."""
    band1 = fitting_mask & (SPA_deg >= BAND1_GAMMA_MIN) & (SPA_deg <= BAND1_GAMMA_MAX)
    band2 = fitting_mask & (SPA_deg >= BAND2_GAMMA_MIN) & (SPA_deg < BAND2_GAMMA_MAX)
    band3 = fitting_mask & (SPA_deg >= BAND3_GAMMA_MIN) & (SPA_deg <= BAND3_GAMMA_MAX)
    band4 = fitting_mask & (ze_deg < BAND4_ZE_MAX)

    multi_band = band1 | band2 | band3 | band4

    return multi_band, {
        'band1_antisolar_primary': band1.sum(),
        'band2_antisolar_secondary': band2.sum(),
        'band3_side': band3.sum(),
        'band4_zenith': band4.sum(),
        'total': multi_band.sum(),
        'bands': {
            'band1': [BAND1_GAMMA_MIN, BAND1_GAMMA_MAX],
            'band2': [BAND2_GAMMA_MIN, BAND2_GAMMA_MAX],
            'band3': [BAND3_GAMMA_MIN, BAND3_GAMMA_MAX],
            'band4_ze_max': BAND4_ZE_MAX,
        }
    }


def build_safe_sampling_mask(clear_sky_mask, disk, ze_deg, sun_exclusion_mask,
                             erosion_iters=SAFE_SAMPLING_EROSION_ITERS,
                             zenith_max_deg=SAFE_SAMPLING_ZENITH_MAX_DEG,
                             min_pixels=SAFE_SAMPLING_MIN_PIXELS):
    """Build a conservative sky-only sampling mask for Chauvin fitting.

    This is designed to be robust even when the input clear_sky_mask is too permissive,
    for example a whole-disk mask on clear days.
    """
    base_mask = clear_sky_mask & disk & (~sun_exclusion_mask)

    eroded_mask = base_mask.copy()
    if erosion_iters > 0:
        eroded_mask = binary_erosion(base_mask, iterations=erosion_iters)

    candidates = [
        ("eroded_75deg", eroded_mask & (ze_deg <= zenith_max_deg)),
        ("base_75deg", base_mask & (ze_deg <= zenith_max_deg)),
        ("base_80deg", base_mask & (ze_deg <= 80.0)),
        ("legacy_85deg", base_mask & (ze_deg <= ZE_HORIZON_CUTOFF)),
    ]

    for label, mask in candidates:
        if mask.sum() >= min_pixels:
            return mask, {
                'label': label,
                'pixels': int(mask.sum()),
                'zenith_max_deg': float(zenith_max_deg if '75deg' in label else (80.0 if '80deg' in label else ZE_HORIZON_CUTOFF)),
                'erosion_iters': int(erosion_iters if label.startswith('eroded') else 0),
            }

    label, mask = candidates[-1]
    return mask, {
        'label': label,
        'pixels': int(mask.sum()),
        'zenith_max_deg': float(ZE_HORIZON_CUTOFF),
        'erosion_iters': 0,
    }



def detect_sun_mask_from_rgb(rgb8, disk_mask, cloud_mask=None, percentile=99.6, min_area=40):
    """Detect sun pixels directly from RGB (independent of semantic annotations)."""

    if cloud_mask is None:
        cloud_mask = np.zeros(rgb8.shape[:2], dtype=bool)

    # Use non-cloud disk pixels to avoid haze/ground artifacts.
    search_mask = disk_mask & (~cloud_mask)
    if not np.any(search_mask):
        search_mask = disk_mask

    # Brightness in linear RGB space is a good stable sun proxy.
    rgb_lin = srgb_to_linear(rgb8.astype(np.float32) / 255.0)
    luminance = 0.2126 * rgb_lin[..., 0] + 0.7152 * rgb_lin[..., 1] + 0.0722 * rgb_lin[..., 2]

    threshold = float(np.percentile(luminance[search_mask], percentile)) if np.any(search_mask) else float(np.percentile(luminance, percentile))
    candidate_mask = (luminance >= threshold) & search_mask

    # Small cleanup to remove noise and connect bright core pixels.
    kernel = np.ones((3, 3), dtype=np.uint8)
    candidate_mask_u8 = (candidate_mask.astype(np.uint8) * 255)
    candidate_mask_u8 = cv2.morphologyEx(candidate_mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    candidate_mask_u8 = cv2.morphologyEx(candidate_mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask_u8, connectivity=8)
    best_label = 0
    best_area = 0
    for lbl in range(1, num_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        if area > best_area:
            best_area = area
            best_label = lbl

    if best_label != 0:
        return labels == best_label

    # Fallback: use brightest pixel inside search region.
    if np.any(search_mask):
        idx = np.argmax(np.where(search_mask, luminance, -1.0))
    else:
        idx = int(np.argmax(luminance))
    y, x = np.unravel_index(idx, luminance.shape)
    fallback = np.zeros_like(search_mask, dtype=np.uint8)
    cv2.circle(fallback, (x, y), 3, 1, -1)
    return fallback.astype(bool)

# ==================== MAIN OPTIMIZER ====================

def optimize_realistic_sun(image_path, mask_path, output_dir):
    """
    Realistic sun blending: Keep real sun pixels + synthetic sky.
    """
    print("="*80)
    print("REALISTIC SUN MODEL OPTIMIZER V2")
    print("Strategy: Real Sun (preserved) + Synthetic Sky (Chauvin) + Smooth Blend")
    print("="*80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load
    print(f"\n[Loading Image]")
    img = Image.open(image_path).convert("RGB")
    rgb8 = np.array(img, np.uint8)
    H, W = rgb8.shape[:2]
    rgb = rgb8.astype(np.float32) / 255.0
    print(f"  Size: {W}×{H}")

    class_masks = load_semantic_mask(mask_path)
    clear_sky_mask_annotation = class_masks.get("clear_sky", np.zeros((H, W), dtype=bool))
    cloud_mask = class_masks.get("cloud", np.zeros((H, W), dtype=bool))
    sun_mask_semantic = class_masks.get("sun", np.zeros((H, W), dtype=bool))
    sun_mask_orig = np.zeros((H, W), dtype=bool)

    print(f"  Clear-sky annotation: {clear_sky_mask_annotation.sum()} px")
    print(f"  Cloud: {cloud_mask.sum()} px")
    print(f"  Semantic sun: {sun_mask_semantic.sum()} px")

    # Prefer user-provided semantic sun mask when available.
    # This keeps sun exclusion anchored to annotation if provided, fallback to RGB.
    if sun_mask_semantic.any():
        sun_mask_orig = sun_mask_semantic.copy()
        sun_source = "semantic"
    else:
        sun_source = "rgb"


    # Geometry
    rgb_lin = srgb_to_linear(rgb)
    sky_combined = clear_sky_mask_annotation | cloud_mask
    if not sky_combined.any():
        # Fallback to luminance mask if semantic annotation is empty
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        th = np.percentile(gray, 99.0)
        sky_combined = gray >= th
    cx, cy, R = fit_disk(sky_combined.astype(np.uint8) * 255)

    # Build geometry first, then detect sun position.
    # Sun source is semantic mask when available, else RGB fallback.
    proj_type = "equisolid"
    theta, ze_deg, az_nav, disk = build_geometry(H, W, cx, cy, R, proj_type, K=1.4)

    if not sun_mask_orig.any():
        sun_mask_orig = detect_sun_mask_from_rgb(rgb8, disk_mask=disk, cloud_mask=cloud_mask)
        sun_source = "rgb"

    class_masks["sun"] = sun_mask_orig

    if sun_mask_orig.any():
        ys_sun, xs_sun = np.nonzero(sun_mask_orig)
        sun_x, sun_y = xs_sun.mean(), ys_sun.mean()
    else:
        sun_x, sun_y = cx, cy

    print(f"  Sun source: {sun_source}")
    print(f"  Sun: {sun_mask_orig.sum()} px")
    print(f"  Sun center: ({sun_x:.1f}, {sun_y:.1f})")

    
    sun_y_int, sun_x_int = int(sun_y), int(sun_x)
    sun_az_nav = az_nav[sun_y_int, sun_x_int]
    sun_ze_deg = ze_deg[sun_y_int, sun_x_int]
    SPA_deg = compute_sun_pixel_angle(az_nav, ze_deg, sun_az_nav, sun_ze_deg)
    gamma = np.radians(SPA_deg)
    
    # Sun exclusion: wider for sky fitting (don't let aureole contaminate sky model)
    EXCLUSION_RADIUS = float(os.environ.get("SUN_EXCLUSION_RADIUS_DEG", SUN_EXCLUSION_RADIUS_DEG))
    sun_exclusion_mask = (SPA_deg <= EXCLUSION_RADIUS) & disk
    print(f"  Sky fitting excludes γ<{EXCLUSION_RADIUS}° ({sun_exclusion_mask.sum():,} px)")
    
    # === STAGE 1: GRID SEARCH WITH MULTI-BAND (Sun Excluded) ===
    print("\n" + "="*80)
    print("STAGE 1: GRID SEARCH WITH MULTI-BAND STRATEGY (Sun Excluded)")
    print("="*80)
    
    safe_sampling_zenith_max_deg = float(os.environ.get("SAFE_SAMPLING_ZENITH_MAX_DEG", SAFE_SAMPLING_ZENITH_MAX_DEG))
    safe_sampling_erosion_iters = int(os.environ.get("SAFE_SAMPLING_EROSION_ITERS", SAFE_SAMPLING_EROSION_ITERS))
    safe_sampling_min_pixels = int(os.environ.get("SAFE_SAMPLING_MIN_PIXELS", SAFE_SAMPLING_MIN_PIXELS))

    # Use geometry only for sampling domain (no clear-sky semantic mask for sampling weights).
    sampling_mask_domain = (ze_deg <= safe_sampling_zenith_max_deg) & disk
    if cloud_mask.any():
        sampling_mask_domain &= ~cloud_mask

    fitting_mask_sky, sampling_meta = build_safe_sampling_mask(
        sampling_mask_domain, disk, ze_deg, sun_exclusion_mask,
        erosion_iters=safe_sampling_erosion_iters,
        zenith_max_deg=safe_sampling_zenith_max_deg,
        min_pixels=safe_sampling_min_pixels,
    )
    clear_sky_mask = fitting_mask_sky
    class_masks["clear_sky"] = clear_sky_mask
    print(f"  Safe sampling mask: {sampling_meta['label']} | zenith≤{sampling_meta['zenith_max_deg']:.1f}° | erosion={sampling_meta['erosion_iters']} px-ish")
    print(f"  Safe sampling pixels: {fitting_mask_sky.sum():,}")
    
    # Prepare both single-band and multi-band masks
    single_band = fitting_mask_sky & (SPA_deg >= BAND_GAMMA_MIN) & (SPA_deg <= BAND_GAMMA_MAX)
    multi_band, band_stats = create_multi_band_mask(SPA_deg, ze_deg, fitting_mask_sky)
    
    print(f"  Single-band ({BAND_GAMMA_MIN:.1f}-{BAND_GAMMA_MAX:.1f}°): {single_band.sum():,} px")
    print(f"  Multi-band: {band_stats['total']:,} px (4 bands)")
    print("  Band windows: "
          f"1[{BAND1_GAMMA_MIN:.1f},{BAND1_GAMMA_MAX:.1f}], "
          f"2[{BAND2_GAMMA_MIN:.1f},{BAND2_GAMMA_MAX:.1f}], "
          f"3[{BAND3_GAMMA_MIN:.1f},{BAND3_GAMMA_MAX:.1f}], "
          f"4<ζ={BAND4_ZE_MAX:.1f}")
    
    fit_method = "per_channel"
    test_sun_sizes = [(0.8, 0.9), (1.0, 1.0), (1.4, 0.8), (0.7, 1.2)]
    
    results = []
    
    # Test both single-band and multi-band for each sun size
    for E_scale, F_scale in test_sun_sizes:
        for band_strategy in ["single", "multi"]:
            band_name = "single-band" if band_strategy == "single" else "multi-band"
            
            # Select band mask
            if band_strategy == "single":
                band_mask = single_band if single_band.sum() >= 50 else fitting_mask_sky
            else:
                band_mask = multi_band if multi_band.sum() >= 50 else fitting_mask_sky
            
            # Fit
            coeffs_dict = fit_per_channel(rgb_lin, theta, gamma, fitting_mask_sky, band_mask)
            
            if coeffs_dict is None:
                continue
            
            rgb_syn, rgb_syn_lin = generate_synthetic_grid(
                coeffs_dict, theta, gamma, disk, E_scale, F_scale, blue_boost=1.0
            )
            
            clearsky_error, sun_error, combined_error = evaluate_synthetic(
                rgb_syn_lin, rgb, clear_sky_mask, sun_mask_orig, disk, ze_deg, cloud_mask=cloud_mask
            )
            
            if np.isnan(combined_error):
                continue
            
            results.append({
                'band_strategy': band_strategy,
                'band_name': band_name,
                'E_scale': E_scale,
                'F_scale': F_scale,
                'coeffs': coeffs_dict,
                'error': combined_error,
            })
            
            print(f"  E×{E_scale}, {band_name}: {combined_error:.4f}")
    
    if not results:
        print("WARN: No band-strategy results; fallback to full fitting mask")
        for E_scale, F_scale in test_sun_sizes:
            band_name = "full"
            coeffs_dict = fit_per_channel(rgb_lin, theta, gamma, fitting_mask_sky, fitting_mask_sky)
            if coeffs_dict is None:
                print(f"  Fallback E×{E_scale}, {band_name}: no coefficients")
                continue

            rgb_syn, rgb_syn_lin = generate_synthetic_grid(
                coeffs_dict, theta, gamma, disk, E_scale, F_scale, blue_boost=1.0
            )
            clearsky_error, sun_error, combined_error = evaluate_synthetic(
                rgb_syn_lin, rgb, clear_sky_mask, sun_mask_orig, disk, ze_deg, cloud_mask=cloud_mask
            )
            if np.isnan(combined_error):
                continue

            results.append({
                'band_strategy': "full",
                'band_name': band_name,
                'E_scale': E_scale,
                'F_scale': F_scale,
                'coeffs': coeffs_dict,
                'error': combined_error,
            })
            print(f"  Fallback E×{E_scale}, {band_name}: {combined_error:.4f}")

        if not results:
            print("ERROR: Grid search failed!")
            return
    
    # Color-aware band selection (prefer single-band when close)
    print(f"\n🎨 COLOR-AWARE BAND SELECTION")
    
    by_sun_size = {}
    for r in results:
        key = r['E_scale']
        if key not in by_sun_size:
            by_sun_size[key] = {'single': None, 'multi': None}
        if r['band_strategy'] == 'single':
            by_sun_size[key]['single'] = r
        else:
            by_sun_size[key]['multi'] = r
    
    adjusted_results = []
    for sun_key, bands in by_sun_size.items():
        single = bands['single']
        multi = bands['multi']
        
        if single and multi:
            # Compare clear-sky errors
            tolerance = 0.15
            if single['error'] <= multi['error'] * (1 + tolerance):
                chosen = single
                reason = f"single-band ({single['error']:.4f} within {tolerance*100:.0f}% of multi {multi['error']:.4f})"
            else:
                chosen = multi
                reason = f"multi-band (better: {multi['error']:.4f} vs {single['error']:.4f})"
            print(f"   E×{sun_key}: {reason}")
            adjusted_results.append(chosen)
        elif single:
            adjusted_results.append(single)
            print(f"   E×{sun_key}: single-band only")
        elif multi:
            adjusted_results.append(multi)
            print(f"   E×{sun_key}: multi-band only")
    
    best_grid = min(adjusted_results, key=lambda x: x['error'])
    print(f"\n✅ Best: E×{best_grid['E_scale']}, {best_grid['band_name']}, Error={best_grid['error']:.4f}")
    
    # Generate best sky
    coeffs_dict = best_grid['coeffs']
    E_scale, F_scale = best_grid['E_scale'], best_grid['F_scale']
    rgb_syn_sky, rgb_syn_sky_lin = generate_synthetic_grid(
        coeffs_dict, theta, gamma, disk, E_scale, F_scale, blue_boost=1.0
    )
    
    # === STAGE 1B: CONTINUOUS REFINEMENT ===
    print("\n" + "="*80)
    print("STAGE 1B: CONTINUOUS REFINEMENT")
    print("="*80)
    
    # Extract parameters
    p0 = []
    for ch in ['R', 'G', 'B']:
        p0.extend(coeffs_dict['coeffs'][ch])
    p0 = np.array(p0)
    
    print(f"  Initial error: {best_grid['error']:.4f}")
    
    refiner = ContinuousRefiner(
        rgb, class_masks,
        theta, gamma, disk, ze_deg, fit_method, sun_exclusion_mask,
        sampling_mask=fitting_mask_sky,
    )
    
    print(f"  Optimizing {len(p0)} parameters...")
    result_refine = refiner.refine(p0, E_scale=E_scale, F_scale=F_scale, maxiter=100)
    
    refined_error = result_refine.fun
    print(f"  Refined error: {refined_error:.4f}")
    
    if refined_error < best_grid['error']:
        improvement = (best_grid['error'] - refined_error) / best_grid['error'] * 100
        print(f"  🎉 Improvement: {improvement:.1f}%")
        refined_params = result_refine.x
    else:
        print(f"  ⚠ No improvement, keeping grid result")
        refined_params = p0
        refined_error = best_grid['error']
    
    # Generate final refined sky
    rgb_syn_sky_lin = refiner.generate_synthetic(refined_params, E_scale=E_scale, F_scale=F_scale)
    rgb_syn_sky = linear_to_srgb(rgb_syn_sky_lin)
    
    print(f"  ✅ Refined Chauvin sky model complete")
    
    # === STAGE 2: REALISTIC SUN BLENDING ===
    print("\n" + "="*80)
    print("STAGE 2: BLEND REAL SUN + SYNTHETIC SKY (Clouds Excluded)")
    print("="*80)
    
    blender = RealisticSunBlender(rgb_lin, SPA_deg, sun_mask_orig, disk, cloud_mask)
    final_synthetic, blend_weights = blender.blend(rgb_syn_sky_lin)
    
    print(f"  ✅ Realistic sun blending complete")
    
    # === EVALUATION ===
    clearsky_error, sun_error, combined_error = evaluate_synthetic(
        final_synthetic, rgb, clear_sky_mask, sun_mask_orig, disk, ze_deg, cloud_mask=cloud_mask
    )
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"  Combined error: {combined_error:.4f}")
    print(f"  Clear-sky error: {clearsky_error:.4f}")
    print(f"  Sun error: {sun_error:.4f}")
    
    # === SAVE OUTPUTS ===
    print("\n[Generating Outputs]")
    
    rgb_syn_srgb = linear_to_srgb(final_synthetic)
    
    rgb_syn_final = rgb_syn_srgb.copy()
    rgb_syn_final[~disk] = 0.0
    rgb_syn_u8 = (np.clip(rgb_syn_final, 0, 1) * 255).astype(np.uint8)
    rgb_syn_u8 = circular_mask(rgb_syn_u8)
    
    syn_file = output_dir / "best_synthetic_clearsky.png"
    Image.fromarray(rgb_syn_u8).save(syn_file)
    print(f"✅ Saved: {syn_file}")
    
    # Results JSON
    results_file = output_dir / "optimization_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'optimization_mode': 'realistic_sun_v2',
            'architecture': 'Real Sun (preserved) + Chauvin Sky + Smooth Transition Blending',
            'sun_strategy': 'Keep real pixels in core, smooth blend to synthetic sky',
            'fitting_strategy': {
                'band_strategy': best_grid['band_name'],
                'E_scale': float(best_grid['E_scale']),
                'F_scale': float(best_grid['F_scale']),
                'grid_error': float(best_grid['error']),
                'refined_error': float(refined_error),
                'improvement_pct': float((best_grid['error'] - refined_error) / best_grid['error'] * 100) if refined_error < best_grid['error'] else 0.0,
            },
            'sampling_strategy': {
                'label': sampling_meta['label'],
                'safe_sampling_pixels': int(fitting_mask_sky.sum()),
                'zenith_max_deg': float(sampling_meta['zenith_max_deg']),
                'erosion_iters': int(sampling_meta['erosion_iters']),
                'single_band_pixels': int(single_band.sum()),
                'multi_band_pixels': int(band_stats['total']),
                'band_breakdown': {
                    'band1_antisolar_primary': int(band_stats['band1_antisolar_primary']),
                    'band2_antisolar_secondary': int(band_stats['band2_antisolar_secondary']),
                    'band3_side': int(band_stats['band3_side']),
                    'band4_zenith': int(band_stats['band4_zenith']),
                    'total': int(band_stats['total'])
                },
                'band_windows': band_stats.get('bands', {}),
            },
            'final_result': {
                'final_error': float(combined_error),
                'clearsky_error': float(clearsky_error),
                'sun_error': float(sun_error),
            },
            'coefficients': {k: ([float(x) for x in v] if isinstance(v, (list, np.ndarray)) else str(v))
                            for k, v in best_grid.get('coeffs', {}).items()},
            'refined_params_flat': [float(x) for x in refined_params] if 'refined_params' in dir() else [],
            'projection': 'equisolid',
            'disk': {'cx': float(cx), 'cy': float(cy), 'R': float(R)},
            'sun_position': {'ze_deg': float(sun_ze_deg), 'az_deg': float(sun_az_nav)},
            'sun_exclusion_radius_deg': float(EXCLUSION_RADIUS),
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)
    print(f"✅ Saved: {results_file}")
    
    # Comparison figure
    comp_file = output_dir / "best_comparison.png"
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Real
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Real Sky", fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Chauvin sky only
    sky_only_srgb = linear_to_srgb(rgb_syn_sky_lin)
    sky_only_srgb[~disk] = 0.0
    sky_only_u8 = (np.clip(sky_only_srgb, 0, 1) * 255).astype(np.uint8)
    sky_only_u8 = circular_mask(sky_only_u8)
    axes[0, 1].imshow(sky_only_u8)
    axes[0, 1].set_title(f"Chauvin Sky Only\n(Sun Excluded)", fontsize=14)
    axes[0, 1].axis('off')
    
    # Real sun region
    sun_region_vis = rgb.copy()
    sun_region_vis[blend_weights < 0.5] *= 0.3  # Dim sky
    axes[0, 2].imshow(sun_region_vis)
    axes[0, 2].set_title(f"Real Sun Region\n(Preserved)", fontsize=14)
    axes[0, 2].axis('off')
    
    # Final blended result
    axes[1, 0].imshow(rgb_syn_u8)
    axes[1, 0].set_title(f"FINAL (Real Sun + Sky)\nError: {combined_error:.4f}", 
                        fontsize=14, fontweight='bold', color='green')
    axes[1, 0].axis('off')
    
    # Blend weights visualization
    blend_vis = plt.cm.RdYlBu_r(blend_weights)[:, :, :3]
    blend_vis = (blend_vis * 255).astype(np.uint8)
    blend_vis = circular_mask(blend_vis)
    axes[1, 1].imshow(blend_vis)
    axes[1, 1].set_title(f"Blend Weights\nRed=Real, Blue=Sky", fontsize=14)
    axes[1, 1].axis('off')
    
    # Difference
    diff = np.abs(rgb - (rgb_syn_u8.astype(np.float32)/255.0))
    diff[~disk] = 0.0
    axes[1, 2].imshow(diff, cmap='hot', vmin=0, vmax=0.3)
    axes[1, 2].set_title(f"Difference\nCS: {clearsky_error:.4f}, Sun: {sun_error:.4f}", fontsize=14)
    axes[1, 2].axis('off')
    
    fig.suptitle(f"Realistic Sun Optimizer V2: Real Sun Preserved + Smooth Blending", 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(comp_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved: {comp_file}")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Realistic Sun Model Optimizer V2")
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    optimize_realistic_sun(args.image, args.mask, args.output)

if __name__ == "__main__":
    main()

