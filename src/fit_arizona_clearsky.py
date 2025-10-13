#!/usr/bin/env python3
"""
Fit clear-sky coefficients for Arizona camera.

Method: Per-channel RGB fitting (no artifacts observed)
Projection: Equisolid angle
Sun size: E×1.0, F×1.0 (baseline)
"""

from pathlib import Path
import json
import numpy as np
from PIL import Image
import sys
from scipy.ndimage import binary_erosion
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from colorspace_utils import srgb_to_linear

# Default paths (can be overridden via command line)
DEFAULT_IMAGE = Path("/Users/max/Desktop/2025-02-24_11_14_00_rgb.webp")
DEFAULT_MASK = Path("/Users/max/Desktop/2025-02-24_11_14_00_rgb_mask_224px.png")
DEFAULT_OUTPUT = Path(__file__).parent.parent / "output/arizona/coefficients.json"

BETA = 0.32
ZE_HORIZON_CUTOFF = 85.0
BAND_GAMMA_MIN = 88.0
BAND_GAMMA_MAX = 92.0

CLASS_COLORS = {
    'background':  (66, 245, 84),
    'clear_sky':   (66, 135, 245),
    'cloud':       (245, 66, 66),
    'sun':         (245, 212, 66),
}

def equisolid_theta_from_radius(r, R):
    """Equisolid angle projection: r = 2R × sin(θ/2)"""
    return 2.0 * np.arcsin(np.clip(r / (2.0 * R), 0, 1))

def compute_sun_pixel_angle(az_pix, ze_pix, az_sun, ze_sun):
    az_pix_rad = np.radians(az_pix)
    ze_pix_rad = np.radians(ze_pix)
    az_sun_rad = np.radians(az_sun)
    ze_sun_rad = np.radians(ze_sun)
    cos_spa = (np.sin(ze_pix_rad) * np.sin(ze_sun_rad) * np.cos(az_pix_rad - az_sun_rad) +
               np.cos(ze_pix_rad) * np.cos(ze_sun_rad))
    cos_spa = np.clip(cos_spa, -1.0, 1.0)
    return np.degrees(np.arccos(cos_spa))

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

def G_model(theta_rad, A, B, C, beta=BETA):
    ct = np.clip(np.cos(theta_rad), 0.0, 1.0)
    return A * (1.0 + C * np.power(ct, beta)) / (1.0 + B * ct + 1e-8)

def S_model(gamma_rad, D, E, F, H):
    g = np.clip(gamma_rad, 1e-3, np.pi)
    return D + E * np.power(g, -F) + H * np.cos(gamma_rad)

def main(image_path=None, mask_path=None, output_path=None):
    image_path = Path(image_path) if image_path else DEFAULT_IMAGE
    mask_path = Path(mask_path) if mask_path else DEFAULT_MASK
    output_path = Path(output_path) if output_path else DEFAULT_OUTPUT
    
    print("="*70)
    print("FITTING ARIZONA CLEAR-SKY COEFFICIENTS")
    print("="*70)
    print(f"Method: Per-channel RGB (no artifacts)")
    print(f"Projection: Equisolid angle")
    print()
    
    # Load image
    img = Image.open(image_path).convert("RGB")
    rgb8 = np.array(img, np.uint8)
    H, W = rgb8.shape[:2]
    rgb_real = rgb8.astype(np.float32) / 255.0
    real_lin = srgb_to_linear(rgb_real)
    
    print(f"Image: {W}×{H}")
    
    # Load mask
    mask_img = Image.open(mask_path).convert("RGB")
    mask_arr = np.array(mask_img, np.uint8)
    clear_sky_mask = np.all(mask_arr == CLASS_COLORS['clear_sky'], axis=-1)
    sun_mask = np.all(mask_arr == CLASS_COLORS['sun'], axis=-1)
    cloud_mask = np.all(mask_arr == CLASS_COLORS['cloud'], axis=-1)
    
    print(f"Clear-sky: {clear_sky_mask.sum():,} pixels")
    print(f"Sun: {sun_mask.sum():,} pixels")
    
    # Geometry
    ys_sun, xs_sun = np.nonzero(sun_mask)
    sun_x, sun_y = xs_sun.mean(), ys_sun.mean()
    
    sky_combined = clear_sky_mask | sun_mask | cloud_mask
    cx, cy, R = fit_disk(sky_combined.astype(np.uint8) * 255)
    print(f"Disk: center=({cx:.1f}, {cy:.1f}), R={R:.1f}px")
    
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    x = xx - cx
    y = cy - yy
    r = np.hypot(x, y)
    r_clipped = np.minimum(r, R)
    
    theta = equisolid_theta_from_radius(r_clipped, R)
    ze_deg = np.degrees(theta)
    phi = np.arctan2(y, x)
    az_math = (np.degrees(phi) % 360.0).astype(np.float32)
    az_nav = (90.0 - az_math) % 360.0
    
    sun_az_nav = az_nav[int(sun_y), int(sun_x)]
    sun_ze_deg = ze_deg[int(sun_y), int(sun_x)]
    print(f"Sun: az={sun_az_nav:.1f}° (nav), ze={sun_ze_deg:.1f}°")
    
    SPA_deg = compute_sun_pixel_angle(az_nav, ze_deg, sun_az_nav, sun_ze_deg)
    gamma = np.radians(SPA_deg)
    disk = r <= R
    
    # Build fitting masks
    non_horizon = (ze_deg <= ZE_HORIZON_CUTOFF)
    fitting_mask = clear_sky_mask & disk & non_horizon
    band = fitting_mask & (SPA_deg >= BAND_GAMMA_MIN) & (SPA_deg <= BAND_GAMMA_MAX)
    
    print(f"Fitting pixels: {fitting_mask.sum():,}")
    
    # Fit per-channel coefficients
    print("\n[Fitting Per-Channel RGB Coefficients]")
    coeffs = {}
    
    for i, ch_name in enumerate(["R", "G", "B"]):
        I = real_lin[..., i]
        
        # Use band if available, otherwise all fitting pixels
        if band.sum() > 100:
            th_band = theta[band]
            I_band = I[band]
        else:
            th_band = theta[fitting_mask]
            I_band = I[fitting_mask]
        
        # Gradation fit
        A0, B0, C0 = max(np.median(I_band), 1e-3), -0.5, 0.2
        bounds_G = ([0.0, -2.0, -2.0], [np.inf, 0.0, 1.5])
        
        def resid_G(params):
            A, B, C = params
            return (G_model(th_band, A, B, C) - I_band)
        
        sol_G = least_squares(resid_G, x0=[A0, B0, C0], bounds=bounds_G, loss='huber', f_scale=0.02)
        A, B, C = sol_G.x
        
        # Scattering fit
        G_full = G_model(theta, A, B, C)
        S_samples = np.clip(I / np.where(G_full <= 1e-8, 1e-8, G_full), 1e-6, None)
        
        g_fit = gamma[fitting_mask]
        s_fit = S_samples[fitting_mask]
        mask_s = (np.degrees(g_fit) > 5.0)
        g_fit = g_fit[mask_s]
        s_fit = s_fit[mask_s]
        
        if g_fit.size < 50:
            print(f"  {ch_name}: Warning - few scattering pixels ({g_fit.size})")
            continue
        
        D0, E0, F0, H0 = float(np.percentile(s_fit, 10)), 1.0, 1.2, 0.2
        bounds_S = ([0.0, 0.0, 0.2, -1.0], [10.0, 1e4, 3.0, 1.0])
        
        def resid_S(params):
            D, E, F, H = params
            return (S_model(g_fit, D, E, F, H) - s_fit)
        
        sol_S = least_squares(resid_S, x0=[D0, E0, F0, H0], bounds=bounds_S, loss='huber', f_scale=0.02)
        D, E, F, H_coeff = sol_S.x
        
        coeffs[ch_name] = {
            "A": float(A),
            "B": float(B),
            "C": float(C),
            "D": float(D),
            "E": float(E),
            "F": float(F),
            "H": float(H_coeff)
        }
        
        print(f"  {ch_name}: A={A:.4f}, B={B:.3f}, C={C:.3f}")
    
    # Save coefficients
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "camera": "Arizona",
        "projection": "Equisolid angle",
        "method": "per_channel",
        "coefficients": coeffs,
        "disk": {"cx": float(cx), "cy": float(cy), "R": float(R)},
        "sun": {"az_nav": float(sun_az_nav), "ze": float(sun_ze_deg), "x": float(sun_x), "y": float(sun_y)}
    }
    
    output_path.write_text(json.dumps(output_data, indent=2))
    
    print(f"\n✅ Saved coefficients: {output_path}")
    print("\n" + "="*70)
    print("Next: Run generate_arizona_synthetic.py")
    print("="*70)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fit Arizona clear-sky coefficients")
    parser.add_argument("--image", help="Path to RGB image")
    parser.add_argument("--mask", help="Path to semantic mask")
    parser.add_argument("--output", help="Path to output JSON")
    args = parser.parse_args()
    
    main(args.image, args.mask, args.output)

