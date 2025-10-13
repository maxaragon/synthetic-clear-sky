#!/usr/bin/env python3
"""
Generate synthetic clear-sky image for Pyranovision camera.

Uses coefficients from fit_pyranovision_clearsky.py.
Allows tuning sun size (E/F scales) and blue boost.
"""

from pathlib import Path
import json
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from colorspace_utils import srgb_to_linear, linear_to_srgb

# Default paths
DEFAULT_IMAGE = Path("/Users/max/Desktop/pyranovision_rgb.png")
DEFAULT_MASK = Path("/Users/max/Desktop/pyranovision_rgb_mask-4_CORRECTED.png")
DEFAULT_COEFFS = Path(__file__).parent.parent / "output/pyranovision/coefficients.json"
DEFAULT_OUTPUT = Path(__file__).parent.parent / "output/pyranovision"

# Default parameters (optimal from testing)
DEFAULT_E_SCALE = 1.4
DEFAULT_F_SCALE = 0.8
DEFAULT_BLUE_BOOST = 1.0
BLUR_SIGMA = 2.0
BETA = 0.32

CLASS_COLORS = {
    'background':  (66, 245, 84),
    'clear_sky':   (66, 135, 245),
    'cloud':       (245, 66, 66),
    'sun':         (245, 212, 66),
}

def k_tan_theta_from_radius(r, R, K):
    denom = np.tan(K * np.pi / 4.0)
    t = (r / R) * denom
    t = np.clip(t, 0.0, 1e6)
    return (2.0 / K) * np.arctan(t)

def compute_sun_pixel_angle(az_pix, ze_pix, az_sun, ze_sun):
    az_pix_rad = np.radians(az_pix)
    ze_pix_rad = np.radians(ze_pix)
    az_sun_rad = np.radians(az_sun)
    ze_sun_rad = np.radians(ze_sun)
    cos_spa = (np.sin(ze_pix_rad) * np.sin(ze_sun_rad) * np.cos(az_pix_rad - az_sun_rad) +
               np.cos(ze_pix_rad) * np.cos(ze_sun_rad))
    cos_spa = np.clip(cos_spa, -1.0, 1.0)
    return np.degrees(np.arccos(cos_spa))

def G_model(theta_rad, A, B, C, beta=BETA):
    ct = np.clip(np.cos(theta_rad), 0.0, 1.0)
    return A * (1.0 + C * np.power(ct, beta)) / (1.0 + B * ct + 1e-8)

def S_model(gamma_rad, D, E, F, H):
    g = np.clip(gamma_rad, 1e-3, np.pi)
    return D + E * np.power(g, -F) + H * np.cos(gamma_rad)

def main(image_path=None, mask_path=None, coeffs_path=None, output_dir=None,
         E_scale=DEFAULT_E_SCALE, F_scale=DEFAULT_F_SCALE, blue_boost=DEFAULT_BLUE_BOOST):
    
    image_path = Path(image_path) if image_path else DEFAULT_IMAGE
    mask_path = Path(mask_path) if mask_path else DEFAULT_MASK
    coeffs_path = Path(coeffs_path) if coeffs_path else DEFAULT_COEFFS
    output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT
    
    print("="*70)
    print("GENERATING PYRANOVISION SYNTHETIC CLEAR-SKY")
    print("="*70)
    print(f"E_scale: {E_scale} (sun brightness)")
    print(f"F_scale: {F_scale} (sun size)")
    print(f"Blue boost: {blue_boost}×")
    print()
    
    # Load coefficients
    coeffs = json.loads(coeffs_path.read_text())
    K = coeffs['K']
    Y_coeffs = coeffs['Y']
    rgb_scales = coeffs['RGB_scales']
    disk_info = coeffs['disk']
    sun_info = coeffs['sun']
    
    print(f"Loaded coefficients: K={K:.1f}")
    print(f"  Gradation: A={Y_coeffs['A']:.4f}, B={Y_coeffs['B']:.4f}, C={Y_coeffs['C']:.4f}")
    print(f"  Scattering: D={Y_coeffs['D']:.4f}, E={Y_coeffs['E']:.4f}, F={Y_coeffs['F']:.4f}, H={Y_coeffs['H']:.4f}")
    
    # Load image
    img = Image.open(image_path).convert("RGB")
    rgb8 = np.array(img, np.uint8)
    H, W = rgb8.shape[:2]
    rgb_real = rgb8.astype(np.float32) / 255.0
    real_lin = srgb_to_linear(rgb_real)
    
    # Load mask
    mask_img = Image.open(mask_path).convert("RGB")
    mask_arr = np.array(mask_img, np.uint8)
    clear_sky_mask = np.all(mask_arr == CLASS_COLORS['clear_sky'], axis=-1)
    sun_mask = np.all(mask_arr == CLASS_COLORS['sun'], axis=-1)
    cloud_mask = np.all(mask_arr == CLASS_COLORS['cloud'], axis=-1)
    
    # Geometry
    cx, cy, R = disk_info['cx'], disk_info['cy'], disk_info['R']
    sun_az_nav, sun_ze = sun_info['az_nav'], sun_info['ze']
    
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    x = xx - cx
    y = cy - yy
    r = np.hypot(x, y)
    r_clipped = np.minimum(r, R)
    
    theta = k_tan_theta_from_radius(r_clipped, R, K=K)
    ze_deg = np.degrees(theta)
    phi = np.arctan2(y, x)
    az_math = (np.degrees(phi) % 360.0).astype(np.float32)
    az_nav = (90.0 - az_math) % 360.0
    
    SPA_deg = compute_sun_pixel_angle(az_nav, ze_deg, sun_az_nav, sun_ze)
    gamma = np.radians(SPA_deg)
    disk = r <= R
    
    # Generate synthetic with specified sun size
    A = Y_coeffs['A']
    B = Y_coeffs['B']
    C = Y_coeffs['C']
    D = Y_coeffs['D']
    E = Y_coeffs['E'] * E_scale
    F = Y_coeffs['F'] * F_scale
    H_coeff = Y_coeffs['H']
    
    Y_syn = G_model(theta, A, B, C) * S_model(gamma, D, E, F, H_coeff)
    
    rgb_syn_lin = np.zeros((H, W, 3), dtype=np.float32)
    rgb_syn_lin[..., 0] = Y_syn * rgb_scales['R']
    rgb_syn_lin[..., 1] = Y_syn * rgb_scales['G']
    rgb_syn_lin[..., 2] = Y_syn * rgb_scales['B'] * blue_boost
    
    # Smooth color gradients
    print("Smoothing color gradients...")
    rgb_syn_lin = cv2.GaussianBlur(rgb_syn_lin, (5, 5), sigmaX=BLUR_SIGMA, sigmaY=BLUR_SIGMA)
    rgb_syn_lin[~disk] = 0
    
    # Auto-scale to match clear-sky brightness
    true_clear_sky = clear_sky_mask & disk & (ze_deg <= 85) & (~cloud_mask)
    valid = true_clear_sky
    
    if valid.sum() > 50:
        real_median = np.median(real_lin[valid])
        syn_median = np.median(rgb_syn_lin[valid])
        if syn_median > 1e-6:
            scale = real_median / syn_median
            rgb_syn_lin *= scale
            print(f"Auto-scale factor: {scale:.3f}×")
    
    rgb_syn = linear_to_srgb(rgb_syn_lin)
    
    # Compute errors
    diff = np.abs(real_lin - rgb_syn_lin)
    diff_mag = diff.sum(axis=-1)
    
    clearsky_region = true_clear_sky & (~sun_mask)
    sun_region = sun_mask & disk
    
    if clearsky_region.sum() > 0:
        clearsky_error = diff_mag[clearsky_region].mean()
        print(f"Clear-sky error: {clearsky_error:.6f}")
    
    if sun_region.sum() > 0:
        sun_error = diff_mag[sun_region].mean()
        print(f"Sun error: {sun_error:.6f}")
    
    if clearsky_region.sum() > 0 and sun_region.sum() > 0:
        combined_error = 0.7 * clearsky_error + 0.3 * sun_error
        print(f"Combined error (70/30): {combined_error:.6f}")
    
    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rgb_syn_uint8 = (np.clip(rgb_syn, 0, 1) * 255).astype(np.uint8)
    syn_path = output_dir / "synthetic_clearsky.png"
    Image.fromarray(rgb_syn_uint8).save(syn_path)
    print(f"\n✅ Saved synthetic: {syn_path}")
    
    # Create comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Full images
    axes[0, 0].imshow(rgb_real)
    axes[0, 0].set_title('Real Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(rgb_syn)
    axes[0, 1].set_title(f'Synthetic Clear-Sky\nE×{E_scale}, F×{F_scale}, Boost×{blue_boost:.2f}',
                         fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    diff_rgb = np.abs(rgb_real - rgb_syn)
    axes[0, 2].imshow(diff_rgb * 3)
    axes[0, 2].set_title('Difference ×3', fontsize=14)
    axes[0, 2].axis('off')
    
    # Row 2: Closeups
    cx_int, cy_int = int(cx), int(cy)
    s = 50
    
    axes[1, 0].imshow(rgb_real[cy_int-s:cy_int+s, cx_int-s:cx_int+s])
    axes[1, 0].set_title('Real Zenith (100×100)', fontsize=12)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(rgb_syn[cy_int-s:cy_int+s, cx_int-s:cx_int+s])
    axes[1, 1].set_title('Synthetic Zenith\n(No chromatic artifacts!)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(diff_rgb[cy_int-s:cy_int+s, cx_int-s:cx_int+s] * 5)
    axes[1, 2].set_title('Zenith Difference ×5', fontsize=12)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    comparison_path = output_dir / "comparison.png"
    fig.savefig(comparison_path, dpi=150)
    print(f"✅ Saved comparison: {comparison_path}")
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Pyranovision synthetic clear-sky")
    parser.add_argument("--image", help="Path to RGB image")
    parser.add_argument("--mask", help="Path to semantic mask")
    parser.add_argument("--coeffs", help="Path to coefficients JSON")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--E_scale", type=float, default=DEFAULT_E_SCALE, help="Sun brightness scale")
    parser.add_argument("--F_scale", type=float, default=DEFAULT_F_SCALE, help="Sun size scale")
    parser.add_argument("--blue_boost", type=float, default=DEFAULT_BLUE_BOOST, help="Blue boost factor")
    args = parser.parse_args()
    
    main(args.image, args.mask, args.coeffs, args.output,
         args.E_scale, args.F_scale, args.blue_boost)
