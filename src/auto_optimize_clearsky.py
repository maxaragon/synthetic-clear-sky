#!/usr/bin/env python3
"""
AUTOMATIC CLEAR-SKY OPTIMIZER

Fully automatic pipeline that tests:
1. Multiple projection models (K-tan with K=1.0-1.8, equidistant, equisolid)
2. Multiple fitting methods (per-channel RGB, Y-based achromatic)
3. Multiple sun size configurations (E/F scaling)
4. Multiple color scaling approaches (median, 90th percentile, saturation boost)

Finds the optimal combination minimizing difference between synthetic and real
in both clear-sky and sun regions.

Usage:
    python auto_optimize_clearsky.py --image IMG.png --mask MASK.png --output OUTPUT_DIR
"""

from pathlib import Path
import json
import numpy as np
from PIL import Image
import cv2
import sys
import argparse
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from colorspace_utils import srgb_to_linear, linear_to_srgb

from scipy.optimize import least_squares
from scipy.ndimage import binary_erosion
import matplotlib.pyplot as plt

# ==================== CONFIGURATION ====================

# Projection models to test
PROJECTIONS = [
    ("ktan_1.0", "ktan", 1.0),
    ("ktan_1.2", "ktan", 1.2),
    ("ktan_1.4", "ktan", 1.4),
    ("ktan_1.6", "ktan", 1.6),
    ("ktan_1.8", "ktan", 1.8),
    ("equidistant", "equidistant", None),
    ("equisolid", "equisolid", None),
]

# Fitting methods
FITTING_METHODS = [
    ("per_channel", "RGB per-channel fitting"),
    ("y_based", "Y-based achromatic + RGB scaling"),
    ("constrained_B", "Y-based with B >= -0.5 (prevents negative zenith)"),
]

# Sun size variants (E_scale, F_scale)
SUN_SIZES = [
    (0.5, 1.5), (0.7, 1.2), (1.0, 1.0), 
    (1.4, 0.8), (1.8, 0.6), (2.5, 0.4),
]

# Color scaling approaches (for Y-based method)
COLOR_SCALES = [
    ("median", "Median RGB/Y ratios"),
    ("p75", "75th percentile RGB/Y ratios"),
    ("p90", "90th percentile RGB/Y ratios"),
    ("blue_boost_1.05", "Median + 5% blue boost"),
    ("blue_boost_1.10", "Median + 10% blue boost"),
]

CLASS_FROM_RGB = {
    (66, 245, 84):  "background",
    (66, 135, 245): "clear_sky",
    (245, 66, 66):  "cloud",
    (245, 212, 66): "sun",
}

BETA = 0.32
ZE_HORIZON_CUTOFF = 85.0
BAND_GAMMA_MIN = 88.0
BAND_GAMMA_MAX = 92.0
SAT_THRESH = 0.98

# ==================== GEOMETRY FUNCTIONS ====================

def k_tan_theta_from_radius(r, R, K=1.4):
    denom = np.tan(K * np.pi / 4.0)
    t = (r / R) * denom
    t = np.clip(t, 0.0, 1e6)
    return (2.0 / K) * np.arctan(t)

def equidistant_theta_from_radius(r, R):
    return (r / R) * (np.pi / 2.0)

def equisolid_theta_from_radius(r, R):
    return 2.0 * np.arcsin(np.clip(r / (2.0 * R), 0, 1))

def orthographic_theta_from_radius(r, R):
    """Orthographic projection: r = R × sin(θ)"""
    return np.arcsin(np.clip(r / R, 0, 1))

def circular_mask(image):
    """Apply a perfect circular mask to an image."""
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), min(h, w) // 2, 255, -1)
    if image.ndim == 3:
        return cv2.bitwise_and(image, image, mask=mask)
    return cv2.bitwise_and(image, mask)

def compute_sun_pixel_angle(az_pix, ze_pix, az_sun, ze_sun):
    az_pix_rad = np.radians(az_pix)
    ze_pix_rad = np.radians(ze_pix)
    az_sun_rad = np.radians(az_sun)
    ze_sun_rad = np.radians(ze_sun)
    
    cos_spa = (np.sin(ze_pix_rad) * np.sin(ze_sun_rad) * np.cos(az_pix_rad - az_sun_rad) +
               np.cos(ze_pix_rad) * np.cos(ze_sun_rad))
    cos_spa = np.clip(cos_spa, -1.0, 1.0)
    spa_rad = np.arccos(cos_spa)
    return np.degrees(spa_rad)

def load_semantic_mask(mask_path):
    mask_img = Image.open(mask_path).convert("RGB")
    mask_arr = np.array(mask_img, np.uint8)
    
    class_masks = {}
    for rgb, class_name in CLASS_FROM_RGB.items():
        match = np.all(mask_arr == rgb, axis=-1)
        class_masks[class_name] = match
    
    return class_masks

def fit_disk(mask_or_gray):
    a = mask_or_gray.astype(np.float32)
    if a.ndim == 3:
        a = a.mean(-1)
    
    mask = a > (5.0 if a.max() > 1.5 else 0.02)
    H, W = a.shape
    
    if not mask.any():
        return (W-1)/2.0, (H-1)/2.0, min(W, H)/2.05
    
    ys, xs = np.nonzero(mask)
    cx, cy = xs.mean(), ys.mean()
    
    edge = mask & (~binary_erosion(mask, iterations=1))
    ys_e, xs_e = np.nonzero(edge)
    
    if xs_e.size < 30:
        R = min(W, H)/2.05
    else:
        R = np.median(np.hypot(xs_e - cx, ys_e - cy))
    
    return cx, cy, R

def build_geometry(H, W, cx, cy, R, proj_type, K=1.4):
    """Build (theta, ze_deg, az_nav, SPA_deg, gamma) grids."""
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    x = xx - cx
    y = cy - yy
    r = np.hypot(x, y)
    r_clipped = np.minimum(r, R)
    
    if proj_type == "ktan":
        theta = k_tan_theta_from_radius(r_clipped, R, K=K)
    elif proj_type == "equidistant":
        theta = equidistant_theta_from_radius(r_clipped, R)
    elif proj_type == "equisolid":
        theta = equisolid_theta_from_radius(r_clipped, R)
    elif proj_type == "orthographic":
        theta = orthographic_theta_from_radius(r_clipped, R)
    else:
        raise ValueError(f"Unknown projection: {proj_type}")
    
    ze_deg = np.degrees(theta)
    phi = np.arctan2(y, x)
    az_math = (np.degrees(phi) % 360.0).astype(np.float32)
    az_nav = (90.0 - az_math) % 360.0
    
    disk = (r <= R)
    
    return theta, ze_deg, az_nav, disk

# ==================== MODEL FUNCTIONS ====================

def G_model(theta_rad, A, B, C, beta=BETA):
    ct = np.clip(np.cos(theta_rad), 0.0, 1.0)
    return A * (1.0 + C * np.power(ct, beta)) / (1.0 + B * ct + 1e-8)

def S_model(gamma_rad, D, E, F, H):
    g = np.clip(gamma_rad, 1e-3, np.pi)
    return D + E * np.power(g, -F) + H * np.cos(gamma_rad)

def percent_clip(x, p=98):
    s = np.sort(x.reshape(-1))
    k = int(np.clip(len(s) * p / 100.0, 0, len(s)-1))
    return s[k]

def linear_to_srgb(linear_rgb):
    a = 0.055
    srgb = np.where(linear_rgb <= 0.0031308,
                    linear_rgb * 12.92,
                    (1 + a) * np.power(np.clip(linear_rgb, 0, None), 1/2.4) - a)
    return np.clip(srgb, 0, 1)

# ==================== FITTING FUNCTIONS ====================

def fit_per_channel(rgb_lin, theta, gamma, fitting_mask, band, constrain_B=False):
    """Fit per-channel RGB coefficients.
    
    Args:
        constrain_B: If True, constrain B >= -0.5 to prevent negative zenith
    """
    coeffs = {}
    chan_names = ["R", "G", "B"]
    chans = [rgb_lin[...,0], rgb_lin[...,1], rgb_lin[...,2]]
    
    for ch_name, I in zip(chan_names, chans):
        sat_cut = percent_clip(I[fitting_mask], SAT_THRESH*100.0)
        valid = fitting_mask & (I <= sat_cut)
        
        # Fit G(θ)
        th_band = theta[band]
        I_band = I[band]
        if I_band.size < 50:
            th_band = theta[valid]
            I_band = I[valid]
        
        if I_band.size < 50:
            return None  # Not enough data
        
        A0, B0, C0 = max(np.median(I_band), 1e-3), -0.3 if constrain_B else -0.5, 0.2
        # Constrain B >= -0.5 to prevent negative denominator at zenith if requested
        bounds_G = ([0.0, -0.5, -2.0], [np.inf, 0.0, 1.5]) if constrain_B else ([0.0, -2.0, -2.0], [np.inf, 0.0, 1.5])
        
        def resid_G(params):
            A, B, C = params
            return (G_model(th_band, A, B, C) - I_band)
        
        try:
            sol_G = least_squares(resid_G, x0=[A0, B0, C0], bounds=bounds_G, loss='huber', f_scale=0.02, max_nfev=100)
            A, B, C = sol_G.x.tolist()
        except:
            return None
        
        # Fit S(γ)
        G_full = G_model(theta, A, B, C)
        S_samples = np.clip(I / np.where(G_full <= 1e-8, 1e-8, G_full), 1e-6, None)
        
        g_fit = gamma[valid]
        s_fit = S_samples[valid]
        mask_s = (np.degrees(g_fit) > 5.0)
        g_fit = g_fit[mask_s]
        s_fit = s_fit[mask_s]
        
        if g_fit.size < 50:
            return None
        
        D0, E0, F0, H0 = float(np.percentile(s_fit, 10)), 1.0, 1.2, 0.2
        bounds_S = ([0.0, 0.0, 0.2, -1.0], [10.0, 1e4, 3.0, 1.0])
        
        def resid_S(params):
            D, E, F, H = params
            return (S_model(g_fit, D, E, F, H) - s_fit)
        
        try:
            sol_S = least_squares(resid_S, x0=[D0, E0, F0, H0], bounds=bounds_S, loss='huber', f_scale=0.02, max_nfev=100)
            D, E, F, H_coeff = sol_S.x.tolist()
        except:
            return None
        
        coeffs[ch_name] = [float(A), float(B), float(C), float(D), float(E), float(F), float(H_coeff)]
    
    # Check consistency
    B_vals = [coeffs[ch][1] for ch in chan_names]
    B_range = max(B_vals) - min(B_vals)
    
    return {"type": "per_channel", "coeffs": coeffs, "B_range": B_range}

def fit_y_based(rgb_lin, theta, gamma, fitting_mask, band, percentile=50, constrain_B=False):
    """Fit Y-based achromatic + RGB scaling.
    
    Args:
        constrain_B: If True, constrain B >= -0.5 to prevent negative zenith
    """
    # Convert to Y
    M_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
    
    H, W = rgb_lin.shape[:2]
    XYZ = (rgb_lin.reshape(-1, 3) @ M_XYZ.T).reshape(H, W, 3)
    Y = XYZ[..., 1]
    
    # Fit Y
    sat_cut = percent_clip(Y[fitting_mask], SAT_THRESH*100.0)
    valid = fitting_mask & (Y <= sat_cut)
    
    th_band = theta[band]
    Y_band = Y[band]
    if Y_band.size < 50:
        th_band = theta[valid]
        Y_band = Y[valid]
    
    if Y_band.size < 50:
        return None
    
    A0, B0, C0 = max(np.median(Y_band), 1e-3), -0.3 if constrain_B else -0.5, 0.2
    # Constrain B >= -0.5 to prevent negative denominator at zenith if requested
    bounds_G = ([0.0, -0.5, -2.0], [np.inf, 0.0, 1.5]) if constrain_B else ([0.0, -2.0, -2.0], [np.inf, 0.0, 1.5])
    
    def resid_G(params):
        A, B, C = params
        return (G_model(th_band, A, B, C) - Y_band)
    
    try:
        sol_G = least_squares(resid_G, x0=[A0, B0, C0], bounds=bounds_G, loss='huber', f_scale=0.02, max_nfev=100)
        A, B, C = sol_G.x.tolist()
    except:
        return None
    
    # Fit S(γ)
    G_full = G_model(theta, A, B, C)
    S_samples = np.clip(Y / np.where(G_full <= 1e-8, 1e-8, G_full), 1e-6, None)
    
    g_fit = gamma[valid]
    s_fit = S_samples[valid]
    mask_s = (np.degrees(g_fit) > 5.0)
    g_fit = g_fit[mask_s]
    s_fit = s_fit[mask_s]
    
    if g_fit.size < 50:
        return None
    
    D0, E0, F0, H0 = float(np.percentile(s_fit, 10)), 1.0, 1.2, 0.2
    bounds_S = ([0.0, 0.0, 0.2, -1.0], [10.0, 1e4, 3.0, 1.0])
    
    def resid_S(params):
        D, E, F, H = params
        return (S_model(g_fit, D, E, F, H) - s_fit)
    
    try:
        sol_S = least_squares(resid_S, x0=[D0, E0, F0, H0], bounds=bounds_S, loss='huber', f_scale=0.02, max_nfev=100)
        D, E, F, H_coeff = sol_S.x.tolist()
    except:
        return None
    
    # Estimate RGB scales at given percentile
    rgb_scales = {}
    Y_syn_temp = G_full * S_model(gamma, D, E, F, H_coeff)
    Y_syn_temp[~fitting_mask] = 0
    
    valid_scale = fitting_mask & (Y_syn_temp > 1e-3)
    if valid_scale.sum() < 50:
        return None
    
    for i, ch_name in enumerate(["R", "G", "B"]):
        ch_real = rgb_lin[..., i]
        ratio = ch_real[valid_scale] / Y_syn_temp[valid_scale]
        scale = float(np.percentile(ratio, percentile))
        rgb_scales[ch_name] = scale
    
    return {
        "type": "y_based",
        "Y": {"A": float(A), "B": float(B), "C": float(C), 
              "D": float(D), "E": float(E), "F": float(F), "H": float(H_coeff)},
        "RGB_scales": rgb_scales,
        "B_range": 0.0  # Achromatic
    }

# ==================== GENERATION FUNCTION ====================

def generate_synthetic(coeffs_dict, theta, gamma, disk, E_scale, F_scale, blue_boost=1.0):
    """Generate synthetic with given coefficients and sun size."""
    H, W = theta.shape
    
    if coeffs_dict["type"] == "per_channel":
        coeffs = coeffs_dict["coeffs"]
        rgb_syn_lin = np.zeros((H, W, 3), dtype=np.float32)
        
        for i, ch in enumerate(["R", "G", "B"]):
            A, B, C, D, E, F, H_coeff = coeffs[ch]
            E_scaled = E * E_scale
            F_scaled = F * F_scale
            
            G = G_model(theta, A, B, C)
            S = S_model(gamma, D, E_scaled, F_scaled, H_coeff)
            L = G * S
            rgb_syn_lin[..., i] = L
    
    elif coeffs_dict["type"] == "y_based":
        Y_coeffs = coeffs_dict["Y"]
        rgb_scales = coeffs_dict["RGB_scales"]
        
        A = Y_coeffs['A']
        B = Y_coeffs['B']
        C = Y_coeffs['C']
        D = Y_coeffs['D']
        E = Y_coeffs['E'] * E_scale
        F = Y_coeffs['F'] * F_scale
        H_coeff = Y_coeffs['H']
        
        G = G_model(theta, A, B, C)
        S = S_model(gamma, D, E, F, H_coeff)
        Y_syn = G * S
        
        rgb_syn_lin = np.zeros((H, W, 3), dtype=np.float32)
        rgb_syn_lin[..., 0] = Y_syn * rgb_scales['R']
        rgb_syn_lin[..., 1] = Y_syn * rgb_scales['G']
        rgb_syn_lin[..., 2] = Y_syn * rgb_scales['B'] * blue_boost
    
    # Apply Gaussian blur for smooth color gradients (matches manual optimal)
    rgb_syn_lin = cv2.GaussianBlur(rgb_syn_lin, (5, 5), sigmaX=2.0, sigmaY=2.0)
    rgb_syn_lin[~disk] = 0
    
    rgb_syn = linear_to_srgb(rgb_syn_lin)
    
    return rgb_syn, rgb_syn_lin

# ==================== EVALUATION FUNCTION ====================

def evaluate_synthetic(rgb_syn_lin, rgb_real, clear_sky_mask, sun_mask, disk, ze_deg, cloud_mask=None):
    """Evaluate synthetic vs real in LINEAR space using L1 norm.
    
    This matches the sun size optimization metric:
    - Compute differences in linear RGB space
    - Use L1 norm (sum of abs differences across channels)
    - Use MEAN error (not median)
    - 70% clear-sky weight, 30% sun weight
    
    Args:
        rgb_syn_lin: Synthetic in linear RGB
        rgb_real: Real in sRGB (will be linearized)
        cloud_mask: Optional cloud mask to EXCLUDE from comparison
    
    Returns:
        (clearsky_error, sun_error, combined_error)
    """
    # Build TRUE clear-sky mask (exclude clouds if provided)
    true_clear_sky = clear_sky_mask & disk & (ze_deg <= 85)
    if cloud_mask is not None:
        true_clear_sky = true_clear_sky & (~cloud_mask)
    
    # Convert real to linear
    real_lin = srgb_to_linear(rgb_real)
    
    # Auto-scale using only TRUE clear-sky pixels
    valid = true_clear_sky
    if valid.sum() < 50:
        return np.nan, np.nan, np.nan
    
    real_median = np.median(real_lin[valid])
    syn_median = np.median(rgb_syn_lin[valid])
    
    if syn_median < 1e-6:
        return np.nan, np.nan, np.nan
    
    scale = real_median / syn_median
    rgb_syn_lin_scaled = rgb_syn_lin * scale
    
    # Compute L1 norm differences in LINEAR space
    diff = np.abs(real_lin - rgb_syn_lin_scaled)
    diff_mag = diff.sum(axis=-1)  # L1 norm: sum of abs differences
    
    # Clear-sky error (EXCLUDE clouds and sun)
    clearsky_only = true_clear_sky & (~sun_mask)
    if clearsky_only.sum() > 0:
        clearsky_error = diff_mag[clearsky_only].mean()  # Use MEAN
    else:
        clearsky_error = np.nan
    
    # Sun region error
    sun_region = sun_mask & disk
    if sun_region.sum() > 0:
        sun_error = diff_mag[sun_region].mean()  # Use MEAN
    else:
        sun_error = np.nan
    
    # Combined metric (70% clear-sky, 30% sun)
    if not np.isnan(clearsky_error) and not np.isnan(sun_error):
        combined_error = 0.7 * clearsky_error + 0.3 * sun_error
    else:
        combined_error = np.nan
    
    return clearsky_error, sun_error, combined_error

# ==================== MAIN OPTIMIZATION ====================

def optimize_clearsky(image_path, mask_path, output_dir):
    """Run full optimization pipeline."""
    print("="*80)
    print("AUTOMATIC CLEAR-SKY OPTIMIZER")
    print("="*80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load image and mask
    print(f"\n[Loading Image & Mask]")
    img = Image.open(image_path).convert("RGB")
    rgb8 = np.array(img, np.uint8)
    H, W = rgb8.shape[:2]
    rgb = rgb8.astype(np.float32) / 255.0
    print(f"  Image: {W}×{H}")
    
    class_masks = load_semantic_mask(mask_path)
    clear_sky_mask = class_masks.get("clear_sky", np.zeros((H, W), dtype=bool))
    sun_mask = class_masks.get("sun", np.zeros((H, W), dtype=bool))
    cloud_mask = class_masks.get("cloud", np.zeros((H, W), dtype=bool))
    
    if not clear_sky_mask.any():
        print("ERROR: No clear_sky pixels!")
        return
    
    print(f"  Clear-sky: {clear_sky_mask.sum()} pixels")
    print(f"  Sun: {sun_mask.sum()} pixels")
    print(f"  Clouds: {cloud_mask.sum()} pixels (EXCLUDED from comparison)")
    
    # Detect sun
    print("\n[Detecting Sun]")
    if sun_mask.any():
        ys_sun, xs_sun = np.nonzero(sun_mask)
        sun_x, sun_y = xs_sun.mean(), ys_sun.mean()
    else:
        rgb_lin_temp = srgb_to_linear(rgb)
        brightness = rgb_lin_temp.mean(axis=-1)
        sun_idx = np.unravel_index(brightness.argmax(), brightness.shape)
        sun_y, sun_x = sun_idx
    print(f"  Sun pixel: ({sun_x:.1f}, {sun_y:.1f})")
    
    # Fit disk
    sky_combined = clear_sky_mask | sun_mask | class_masks.get("cloud", np.zeros((H, W), dtype=bool))
    cx, cy, R = fit_disk(sky_combined.astype(np.uint8) * 255)
    print(f"  Disk: center=({cx:.1f}, {cy:.1f}), R={R:.1f}px")
    
    # Convert to linear RGB
    rgb_lin = srgb_to_linear(rgb)
    
    # MAIN OPTIMIZATION LOOP
    results = []
    total_tests = len(PROJECTIONS) * len(FITTING_METHODS) * len(SUN_SIZES) * len(COLOR_SCALES)
    current_test = 0
    
    print(f"\n[Starting Optimization: {total_tests} configurations]")
    print("="*80)
    
    for proj_name, proj_type, K_val in PROJECTIONS:
        print(f"\n📐 Testing projection: {proj_name}")
        
        # Build geometry
        theta, ze_deg, az_nav, disk = build_geometry(H, W, cx, cy, R, proj_type, K=K_val or 1.4)
        
        # Sun angles
        sun_y_int, sun_x_int = int(sun_y), int(sun_x)
        sun_az_nav = az_nav[sun_y_int, sun_x_int]
        sun_ze_deg = ze_deg[sun_y_int, sun_x_int]
        
        SPA_deg = compute_sun_pixel_angle(az_nav, ze_deg, sun_az_nav, sun_ze_deg)
        gamma = np.radians(SPA_deg)
        
        # Build fitting mask
        non_horizon = (ze_deg <= ZE_HORIZON_CUTOFF)
        fitting_mask = clear_sky_mask & disk & non_horizon
        band = fitting_mask & (SPA_deg >= BAND_GAMMA_MIN) & (SPA_deg <= BAND_GAMMA_MAX)
        
        if fitting_mask.sum() < 100:
            print(f"  ⚠️  Skipped: insufficient fitting pixels")
            continue
        
        for fit_method, fit_desc in FITTING_METHODS:
            print(f"  🔧 Fitting method: {fit_method}")
            
            # Fit coefficients
            if fit_method == "per_channel":
                coeffs_dict = fit_per_channel(rgb_lin, theta, gamma, fitting_mask, band, constrain_B=False)
            elif fit_method == "y_based":
                coeffs_dict = fit_y_based(rgb_lin, theta, gamma, fitting_mask, band, percentile=50, constrain_B=False)
            elif fit_method == "constrained_B":
                # Use Y-based fitting with constrained B to prevent negative zenith
                coeffs_dict = fit_y_based(rgb_lin, theta, gamma, fitting_mask, band, percentile=50, constrain_B=True)
            
            if coeffs_dict is None:
                print(f"    ⚠️  Fitting failed")
                continue
            
            print(f"    B range: {coeffs_dict['B_range']:.3f}")
            
            # Test different color scales (only for Y-based)
            color_scale_list = COLOR_SCALES if fit_method == "y_based" else [("default", "Default")]
            
            for color_scale_name, color_scale_desc in color_scale_list:
                # Modify RGB scales if needed
                if fit_method == "y_based" and color_scale_name != "default":
                    if color_scale_name == "p75":
                        coeffs_dict_modified = fit_y_based(rgb_lin, theta, gamma, fitting_mask, band, percentile=75)
                    elif color_scale_name == "p90":
                        coeffs_dict_modified = fit_y_based(rgb_lin, theta, gamma, fitting_mask, band, percentile=90)
                    else:
                        coeffs_dict_modified = coeffs_dict
                    
                    if coeffs_dict_modified is None:
                        continue
                    
                    blue_boost = 1.0
                    if "blue_boost_1.05" in color_scale_name:
                        blue_boost = 1.05
                    elif "blue_boost_1.10" in color_scale_name:
                        blue_boost = 1.10
                else:
                    coeffs_dict_modified = coeffs_dict
                    blue_boost = 1.0
                
                # Test sun sizes
                for E_scale, F_scale in SUN_SIZES:
                    current_test += 1
                    
                    # Generate synthetic
                    rgb_syn, rgb_syn_lin = generate_synthetic(
                        coeffs_dict_modified, theta, gamma, disk, E_scale, F_scale, blue_boost
                    )
                    
                    # Evaluate (EXCLUDE clouds from comparison!)
                    clearsky_error, sun_error, combined_error = evaluate_synthetic(
                        rgb_syn_lin, rgb, clear_sky_mask, sun_mask, disk, ze_deg, cloud_mask=cloud_mask
                    )
                    
                    if np.isnan(combined_error):
                        continue
                    
                    # Store result (convert numpy types to Python types for JSON)
                    result = {
                        'projection': proj_name,
                        'projection_type': proj_type,
                        'K': float(K_val) if K_val is not None else None,
                        'fitting_method': fit_method,
                        'color_scale': color_scale_name,
                        'E_scale': float(E_scale),
                        'F_scale': float(F_scale),
                        'blue_boost': float(blue_boost),
                        'clearsky_error': float(clearsky_error),
                        'sun_error': float(sun_error),
                        'combined_error': float(combined_error),
                        'B_range': float(coeffs_dict_modified['B_range']),
                        'sun_ze': float(sun_ze_deg),
                    }
                    results.append(result)
                    
                    # Progress (show best so far)
                    if current_test % 10 == 0 or combined_error < min([r['combined_error'] for r in results[:-1]], default=999):
                        best_so_far = min(results, key=lambda x: x['combined_error'])
                        print(f"    Progress: {current_test}/{total_tests} | Best: {best_so_far['combined_error']:.4f} [{best_so_far['fitting_method']}, E×{best_so_far['E_scale']}]")
    
    # Find best result
    print("\n" + "="*80)
    print("[Optimization Complete]")
    print(f"  Tested: {len(results)} valid configurations")
    
    if not results:
        print("ERROR: No valid results!")
        return
    
    best = min(results, key=lambda x: x['combined_error'])
    
    print("\n🏆 BEST CONFIGURATION:")
    print(f"  Projection: {best['projection']}")
    print(f"  Fitting: {best['fitting_method']}")
    print(f"  Color scale: {best['color_scale']}")
    print(f"  Sun size: E×{best['E_scale']:.2f}, F×{best['F_scale']:.2f}")
    print(f"  Blue boost: {best['blue_boost']:.2f}×")
    print(f"  ━" * 40)
    print(f"  📊 ERRORS (L1 norm, linear RGB):")
    print(f"     Clear-sky:  {best['clearsky_error']:.6f}")
    print(f"     Sun region: {best['sun_error']:.6f}")
    print(f"     Combined:   {best['combined_error']:.6f} ← (70% sky + 30% sun)")
    print(f"  ━" * 40)
    print(f"  B range: {best['B_range']:.3f}")
    
    # Show top 5 configurations
    print("\n📊 TOP 5 CONFIGURATIONS:")
    top5 = sorted(results, key=lambda x: x['combined_error'])[:5]
    for i, config in enumerate(top5, 1):
        print(f"\n  {i}. {config['fitting_method']} ({config['projection']})")
        print(f"     Sun: E×{config['E_scale']:.1f}, F×{config['F_scale']:.1f} | Boost: {config['blue_boost']:.2f}×")
        print(f"     Clear-sky: {config['clearsky_error']:.6f} | Sun: {config['sun_error']:.6f}")
        print(f"     Combined:  {config['combined_error']:.6f}")
    
    # Save results
    results_file = output_dir / "optimization_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'best': best,
            'all_results': results,
            'timestamp': datetime.now().isoformat(),
            'image': str(image_path),
            'mask': str(mask_path),
        }, f, indent=2)
    
    print(f"\n✅ Saved results: {results_file}")
    
    # Generate best synthetic clear-sky image
    print("\n[Generating Best Synthetic Clear-Sky]")
    print(f"  Configuration: {best['fitting_method']} ({best['projection']})")
    print(f"  Sun size: E×{best['E_scale']:.2f}, F×{best['F_scale']:.2f}")
    print(f"  Blue boost: {best['blue_boost']:.2f}×")
    
    # Regenerate geometry with best projection
    proj_name = best['projection']
    proj_type = None
    proj_K = None
    for pn, pt, pk in PROJECTIONS:
        if pn == proj_name:
            proj_type = pt
            proj_K = pk
            break
    
    # Rebuild geometry
    theta, ze_deg, az_nav, disk = build_geometry(H, W, cx, cy, R, proj_type, K=proj_K or 1.4)
    
    # Sun angles
    sun_y_int, sun_x_int = int(sun_y), int(sun_x)
    sun_az_nav = az_nav[sun_y_int, sun_x_int]
    sun_ze_deg = ze_deg[sun_y_int, sun_x_int]
    
    SPA_deg = compute_sun_pixel_angle(az_nav, ze_deg, sun_az_nav, sun_ze_deg)
    gamma = np.radians(SPA_deg)
    
    # Rebuild fitting mask
    non_horizon = (ze_deg <= ZE_HORIZON_CUTOFF)
    fitting_mask_best = clear_sky_mask & disk & non_horizon
    band = fitting_mask_best & (SPA_deg >= BAND_GAMMA_MIN) & (SPA_deg <= BAND_GAMMA_MAX)
    
    # Refit with best method
    if best['fitting_method'] == "per_channel":
        coeffs_best = fit_per_channel(rgb_lin, theta, gamma, fitting_mask_best, band, constrain_B=False)
    elif best['fitting_method'] == "y_based":
        coeffs_best = fit_y_based(rgb_lin, theta, gamma, fitting_mask_best, band, percentile=50, constrain_B=False)
    elif best['fitting_method'] == "constrained_B":
        coeffs_best = fit_y_based(rgb_lin, theta, gamma, fitting_mask_best, band, percentile=50, constrain_B=True)
    
    # Generate synthetic
    _, rgb_syn_lin = generate_synthetic(coeffs_best, theta, gamma, disk, 
                                       best['E_scale'], best['F_scale'], best['blue_boost'])
    
    # Auto-scale
    real_median = np.array([np.median(rgb_lin[..., i][clear_sky_mask]) for i in range(3)])
    syn_median = np.array([np.median(rgb_syn_lin[..., i][clear_sky_mask]) for i in range(3)])
    scale = real_median / (syn_median + 1e-6)
    for i in range(3):
        rgb_syn_lin[..., i] *= scale[i]
    
    # Convert to sRGB
    rgb_syn_srgb = linear_to_srgb(rgb_syn_lin)
    
    # Apply background mask
    bg_mask = class_masks.get("background", np.zeros((H, W), dtype=bool))
    rgb_syn_final = rgb_syn_srgb.copy()
    rgb_syn_final[bg_mask] = 0.0
    
    # Apply circular mask
    rgb_syn_u8 = (np.clip(rgb_syn_final, 0, 1) * 255).astype(np.uint8)
    rgb_syn_u8 = circular_mask(rgb_syn_u8)
    
    # Save synthetic
    syn_file = output_dir / "best_synthetic_clearsky.png"
    Image.fromarray(rgb_syn_u8).save(syn_file)
    print(f"✅ Saved: {syn_file}")
    
    # Create comparison
    comp_file = output_dir / "best_comparison.png"
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(rgb)
    axes[0].set_title("Real Sky", fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(rgb_syn_u8)
    axes[1].set_title(f"Best Synthetic Clear-Sky\n{best['fitting_method']} ({best['projection']})", fontsize=14)
    axes[1].axis('off')
    
    diff = np.abs(rgb - (rgb_syn_u8.astype(np.float32)/255.0))
    diff_masked = diff.copy()
    diff_masked[bg_mask] = 0.0
    axes[2].imshow(diff_masked, cmap='hot', vmin=0, vmax=0.3)
    axes[2].set_title(f"Difference (L1)\nCombined Error: {best['combined_error']:.3f}", fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    fig.savefig(comp_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved: {comp_file}")
    
    print("="*80)
    return best, results

# ==================== CLI ====================

def main():
    parser = argparse.ArgumentParser(description="Automatic Clear-Sky Optimizer")
    parser.add_argument("--image", required=True, help="Path to RGB sky image")
    parser.add_argument("--mask", required=True, help="Path to semantic mask")
    parser.add_argument("--output", required=True, help="Output directory")
    
    args = parser.parse_args()
    
    optimize_clearsky(args.image, args.mask, args.output)

if __name__ == "__main__":
    main()

