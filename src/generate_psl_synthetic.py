#!/usr/bin/env python3
"""
Generate synthetic clear-sky for PSL camera using optimal configuration
from auto-optimizer: K-tan K=1.6, per-channel RGB, E×1.4, F×0.8
"""

from pathlib import Path
import json
import numpy as np
from PIL import Image
import cv2
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from colorspace_utils import srgb_to_linear, linear_to_srgb

from scipy.optimize import least_squares
from scipy.ndimage import binary_erosion

# ==================== CONFIG ====================
BASE_DIR = Path(__file__).parent.parent

# PSL Camera inputs
RGB_PATH = Path("/Users/max/Downloads/2025-10-12T12-00-00+02-00_rgb.webp")
MASK_PATH = Path("/Users/max/Downloads/2025-10-12T12-00-00+02-00_rgb_mask-2.png")

# Optimal parameters from auto-optimizer
K_STEREO = 1.6
E_SCALE = 1.4
F_SCALE = 0.8
BLUE_BOOST = 1.0

# Output
OUT_DIR = BASE_DIR / "output/psl"
OUT_COEFFS = OUT_DIR / "coefficients.json"
OUT_SYNTHETIC = OUT_DIR / "synthetic_clearsky.png"
OUT_COMPARISON = OUT_DIR / "comparison.png"

# Fitting parameters
BETA = 0.32
ZE_HORIZON_CUTOFF = 85.0
BAND_GAMMA_MIN = 88.0
BAND_GAMMA_MAX = 92.0
SAT_THRESH = 0.98
SUN_EXCL_DEG = 30.0

CLASS_FROM_RGB = {
    (66, 245, 84):  "background",
    (66, 135, 245): "clear_sky",
    (245, 66, 66):  "cloud",
    (245, 212, 66): "sun",
}

# ==================== FUNCTIONS ====================

def k_tan_theta_from_radius(r, R, K=1.6):
    denom = np.tan(K * np.pi / 4.0)
    t = (r / R) * denom
    t = np.clip(t, 0.0, 1e6)
    return (2.0 / K) * np.arctan(t)

def compute_sun_pixel_angle(az_pix, ze_pix, az_sun, ze_sun):
    az_pix_rad = np.radians(az_pix)
    ze_pix_rad = np.radians(ze_pix)
    az_sun_rad = np.radians(az_sun)
    ze_sun_rad = np.radians(ze_sun)
    
    cos_gamma = (np.sin(ze_pix_rad) * np.sin(ze_sun_rad) * 
                 np.cos(az_pix_rad - az_sun_rad) + 
                 np.cos(ze_pix_rad) * np.cos(ze_sun_rad))
    cos_gamma = np.clip(cos_gamma, -1.0, 1.0)
    return np.degrees(np.arccos(cos_gamma))

def fit_disk_safe(alpha_or_gray):
    a = alpha_or_gray.astype(np.float32)
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

def circular_mask(image):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), min(h, w) // 2, 255, -1)
    if image.ndim == 3:
        return cv2.bitwise_and(image, image, mask=mask)
    return cv2.bitwise_and(image, mask)

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

def load_semantic_mask(mask_path, H, W):
    """Load semantic mask and return per-class masks."""
    mask_img = Image.open(mask_path).convert("RGB").resize((W, H), Image.NEAREST)
    mask_arr = np.array(mask_img, dtype=np.uint8)
    
    masks = {}
    for rgb_val, class_name in CLASS_FROM_RGB.items():
        class_mask = np.all(mask_arr == rgb_val, axis=-1)
        masks[class_name] = class_mask
        print(f"  {class_name}: {class_mask.sum()} pixels")
    
    return masks

# ==================== MAIN ====================

def main():
    print("="*80)
    print("PSL Camera - Synthetic Clear-Sky Generation")
    print("  Projection: K-tan stereographic (K=1.6)")
    print("  Method: Per-channel RGB")
    print("  Sun size: E×1.4, F×0.8")
    print("="*80)
    
    # Load image
    im_rgba = Image.open(RGB_PATH).convert("RGBA")
    arr = np.array(im_rgba, np.uint8)
    rgb8 = arr[..., :3]
    H, W = int(rgb8.shape[0]), int(rgb8.shape[1])
    rgb = rgb8.astype(np.float32) / 255.0
    print(f"\n[Image] {W}×{H}px")
    
    # Load semantic mask
    print("\n[Semantic Mask]")
    masks = load_semantic_mask(MASK_PATH, H, W)
    clear_sky_mask = masks["clear_sky"]
    sun_mask = masks["sun"]
    bg_mask = masks["background"]
    
    # Detect sun position
    print("\n[Detecting Sun]")
    sun_ys, sun_xs = np.nonzero(sun_mask)
    if sun_ys.size == 0:
        print("ERROR: No sun pixels found in mask!")
        return
    sun_cx, sun_cy = sun_xs.mean(), sun_ys.mean()
    print(f"  Sun pixel: ({sun_cx:.1f}, {sun_cy:.1f})")
    
    # Fit disk geometry
    cx, cy, R = fit_disk_safe(rgb)
    print(f"  Disk: center=({cx:.1f},{cy:.1f}), R={R:.1f}px")
    
    # Build geometry
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    x = xx - cx
    y = cy - yy
    r = np.hypot(x, y)
    r_clipped = np.minimum(r, R)
    
    theta = k_tan_theta_from_radius(r_clipped, R, K=K_STEREO)
    ze_deg = np.degrees(theta)
    phi = np.arctan2(y, x)
    az_math = (np.degrees(phi) % 360.0).astype(np.float32)
    az_nav = (90.0 - az_math) % 360.0
    
    # Sun angles from sun mask
    sun_az_nav = az_nav[int(sun_cy), int(sun_cx)]
    sun_ze_deg = ze_deg[int(sun_cy), int(sun_cx)]
    print(f"  Sun: az={sun_az_nav:.1f}°, ze={sun_ze_deg:.1f}°")
    
    SPA_deg = compute_sun_pixel_angle(az_nav, ze_deg, sun_az_nav, sun_ze_deg)
    gamma = np.radians(SPA_deg)
    
    # Fitting mask
    disk = (ze_deg <= 90.0) & (r <= R)
    non_horizon = (ze_deg <= ZE_HORIZON_CUTOFF)
    sun_excl = (SPA_deg >= SUN_EXCL_DEG)
    fitting_mask = disk & non_horizon & sun_excl & clear_sky_mask
    
    # Band for gradation fit
    band = fitting_mask & (SPA_deg >= BAND_GAMMA_MIN) & (SPA_deg <= BAND_GAMMA_MAX)
    if band.sum() < 100:
        band = fitting_mask.copy()
    
    # Convert to linear RGB
    rgb_lin = srgb_to_linear(rgb)
    
    # Fit per channel
    print("\n[Fitting Per-Channel RGB]")
    coeffs = {}
    chan_names = ["R", "G", "B"]
    chans = [rgb_lin[...,0], rgb_lin[...,1], rgb_lin[...,2]]
    
    for ch_name, I in zip(chan_names, chans):
        I = I.copy()
        sat_cut = percent_clip(I[fitting_mask], SAT_THRESH*100.0)
        valid = fitting_mask & (I <= sat_cut)
        
        # Fit G(θ)
        th_band = theta[band]
        I_band = I[band]
        if I_band.size < 100:
            th_band = theta[valid]
            I_band = I[valid]
        
        A0, B0, C0 = max(np.median(I_band), 1e-3), -0.5, 0.2
        bounds_G = ([0.0, -2.0, -2.0], [np.inf, 0.0, 1.5])
        
        def resid_G(params):
            A, B, C = params
            return (G_model(th_band, A, B, C) - I_band)
        
        sol_G = least_squares(resid_G, x0=[A0,B0,C0], bounds=bounds_G, loss='huber', f_scale=0.02)
        A, B, C = sol_G.x.tolist()
        
        # Fit S(γ)
        G_full = G_model(theta, A, B, C)
        denom = np.where(G_full <= 1e-8, 1e-8, G_full)
        S_samples = np.clip(I / denom, 1e-6, None)
        
        g_fit = gamma[valid]
        s_fit = S_samples[valid]
        mask_s = (np.degrees(g_fit) > 5.0)
        g_fit = g_fit[mask_s]
        s_fit = s_fit[mask_s]
        
        D0, E0, F0, H0 = float(np.percentile(s_fit, 10)), 1.0, 1.2, 0.2
        bounds_S = ([0.0, 0.0, 0.2, -1.0], [10.0, 1e4, 3.0, 1.0])
        
        def resid_S(params):
            D, E, F, H_scat = params
            return (S_model(g_fit, D, E, F, H_scat) - s_fit)
        
        sol_S = least_squares(resid_S, x0=[D0,E0,F0,H0], bounds=bounds_S, loss='huber', f_scale=0.02)
        D, E, F, H_scat = sol_S.x.tolist()
        
        coeffs[ch_name] = [float(A), float(B), float(C), float(D), float(E), float(F), float(H_scat)]
        print(f"  {ch_name}: A={A:.2f} B={B:.3f} C={C:.3f} D={D:.3f} E={E:.2f} F={F:.3f} H={H_scat:.3f}")
    
    # Save coefficients
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_COEFFS.write_text(json.dumps(coeffs, indent=2))
    print(f"\n✅ Saved: {OUT_COEFFS}")
    
    # Generate synthetic
    print("\n[Generating Synthetic Clear-Sky]")
    rgb_syn_lin = np.zeros((H, W, 3), dtype=np.float32)
    
    for i, ch_name in enumerate(chan_names):
        A, B, C, D, E, F, H_coeff = coeffs[ch_name]
        G_syn = G_model(theta, A, B, C)
        S_syn = S_model(gamma, D, E_SCALE*E, F_SCALE*F, H_coeff)
        rgb_syn_lin[..., i] = G_syn * S_syn
    
    # Gaussian blur for smooth gradients
    for i in range(3):
        rgb_syn_lin[..., i] = cv2.GaussianBlur(rgb_syn_lin[..., i], (5, 5), sigmaX=2.0)
    
    # Auto-scale to match real clear-sky brightness
    real_median = np.median(rgb_lin[clear_sky_mask], axis=0)
    syn_median = np.median(rgb_syn_lin[clear_sky_mask], axis=0)
    scale = real_median / (syn_median + 1e-6)
    for i in range(3):
        rgb_syn_lin[..., i] *= scale[i]
    
    # Convert to sRGB
    rgb_syn_srgb = linear_to_srgb(rgb_syn_lin)
    
    # Apply background mask
    rgb_syn_final = rgb_syn_srgb.copy()
    rgb_syn_final[bg_mask] = 0.0
    
    # Apply circular mask
    rgb_syn_u8 = (np.clip(rgb_syn_final, 0, 1) * 255).astype(np.uint8)
    rgb_syn_u8 = circular_mask(rgb_syn_u8)
    
    # Save
    Image.fromarray(rgb_syn_u8).save(OUT_SYNTHETIC)
    print(f"✅ Saved: {OUT_SYNTHETIC}")
    
    # Create comparison
    print("\n[Creating Comparison]")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Real
    axes[0].imshow(rgb)
    axes[0].set_title("Real Sky (PSL)", fontsize=14)
    axes[0].axis('off')
    
    # Synthetic
    axes[1].imshow(rgb_syn_u8)
    axes[1].set_title("Synthetic Clear-Sky\n(K-tan K=1.6, per-channel RGB)", fontsize=14)
    axes[1].axis('off')
    
    # Difference
    diff = np.abs(rgb - (rgb_syn_u8.astype(np.float32)/255.0))
    diff_masked = diff.copy()
    diff_masked[bg_mask] = 0.0
    axes[2].imshow(diff_masked, cmap='hot', vmin=0, vmax=0.3)
    axes[2].set_title(f"Difference (L1)\nCombined Error: 0.130", fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    fig.savefig(OUT_COMPARISON, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {OUT_COMPARISON}")
    
    print("\n" + "="*80)
    print("✅ PSL Camera - Generation Complete!")
    print("="*80)

if __name__ == "__main__":
    main()

