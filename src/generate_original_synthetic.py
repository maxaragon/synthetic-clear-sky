#!/usr/bin/env python3
"""
Step 2: Generate synthetic clear-sky RGB images using fitted coefficients

Uses the Chauvin et al. model with the fitted per-channel coefficients:
  L_ch(θ,γ) = G_ch(θ) * S_ch(γ)

Loads coefficients from Step 1 and generates synthetic RGB clear-sky images.
You can adjust the sun position to generate clear-sky for any solar angle!

Outputs:
  - ../output/synthetic/synthetic_clearsky.png
  - ../output/synthetic/comparison_real_vs_synthetic.png
"""

from pathlib import Path
import json
import numpy as np
from PIL import Image
import cv2
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from colorspace_utils import srgb_to_linear
from geometry_utils import compute_sun_pixel_angle

from scipy.ndimage import binary_erosion

# ----------------- CONFIG -----------------
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent

REAL_IMG = BASE_DIR / "input/clearsky_128px.png"
COEFFS_JSON = BASE_DIR / "output/parameters/clearsky_RGB_coefficients.json"
OUT_SYNTHETIC = BASE_DIR / "output/synthetic/synthetic_clearsky.png"
OUT_COMPARISON = BASE_DIR / "output/synthetic/comparison_real_vs_synthetic.png"

# Geometry settings (must match unwrap)
K_STEREO = 1.4
BETA = 0.32

# Generation settings
EXPOSURE_SCALE = 1.0      # Adjust to tune brightness
GAMMA_ENCODE = True       # True = output sRGB, False = linear
USE_DETECTED_SUN = True   # True = detect sun from image, False = use manual values below

# Manual sun position (only used if USE_DETECTED_SUN = False)
MANUAL_SUN_AZ_NAV = 180.0  # Azimuth in navigation convention (0°=North, clockwise)
MANUAL_SUN_ZE = 45.0       # Zenith angle in degrees

# ----------------- helpers -----------------
def k_tan_theta_from_radius(r, R, K=1.4):
    """Inverse K-tan stereographic: radius r → zenith θ (radians)."""
    denom = np.tan(K * np.pi / 4.0)
    t = (r / R) * denom
    t = np.clip(t, 0.0, 1e6)
    return (2.0 / K) * np.arctan(t)

def fit_disk_safe(alpha_or_gray):
    """Robust disk center/radius estimation."""
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
    """Perfect central circular mask."""
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), min(h, w) // 2, 255, -1)
    return cv2.bitwise_and(image, image, mask=mask)

def G_model(theta_rad, A, B, C, beta=BETA):
    """Gradation term: zenith angle dependence."""
    ct = np.clip(np.cos(theta_rad), 0.0, 1.0)
    return A * (1.0 + C * np.power(ct, beta)) / (1.0 + B * ct + 1e-8)

def S_model(gamma_rad, D, E, F, H):
    """Scattering term: sun-pixel angle dependence."""
    g = np.clip(gamma_rad, 1e-3, np.pi)
    return D + E * np.power(g, -F) + H * np.cos(gamma_rad)

def linear_to_srgb(linear_rgb):
    """Convert linear RGB to sRGB (gamma encoding)."""
    a = 0.055
    srgb = np.where(linear_rgb <= 0.0031308,
                    linear_rgb * 12.92,
                    (1 + a) * np.power(np.clip(linear_rgb, 0, None), 1/2.4) - a)
    return np.clip(srgb, 0, 1)

# ----------------- main -----------------
def main():
    print("="*70)
    print("STEP 2: Generating synthetic clear-sky RGB image")
    print("="*70)
    
    # Load coefficients
    if not COEFFS_JSON.exists():
        print(f"ERROR: Coefficients not found at {COEFFS_JSON}")
        print("Please run scripts/1_fit_rgb_coefficients.py first!")
        return
    
    coeffs = json.loads(COEFFS_JSON.read_text())
    print(f"Loaded coefficients from {COEFFS_JSON.name}")
    for ch in ["R", "G", "B"]:
        A,B,C,D,E,F,H = coeffs[ch]
        print(f"  {ch}: A={A:.4f} B={B:.4f} C={C:.4f}")
    
    # Load real image for geometry
    if not REAL_IMG.exists():
        print(f"ERROR: Input image not found: {REAL_IMG}")
        return
    
    im_rgba = Image.open(REAL_IMG).convert("RGBA")
    arr = np.array(im_rgba, np.uint8)
    rgb8_real = arr[..., :3]
    alpha = arr[..., 3]
    H, W = alpha.shape
    print(f"\nImage: {W}×{H}")
    
    rgb_real = rgb8_real.astype(np.float32) / 255.0
    disk = alpha > 5
    if not disk.any():
        disk = (rgb_real.mean(-1) > 1/255)
    
    # Rebuild geometry
    cx, cy, R = fit_disk_safe(alpha if (alpha > 0).any() else rgb_real)
    print(f"Disk: center=({cx:.2f},{cy:.2f}) R={R:.2f}px")
    
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
    
    # Sun position
    if USE_DETECTED_SUN:
        print("\n[Detecting Sun from Image]")
        rgb_lin_real = srgb_to_linear(rgb_real)
        brightness = rgb_lin_real.mean(axis=-1)
        brightness_valid = brightness.copy()
        brightness_valid[~disk] = 0
        threshold = np.percentile(brightness_valid[disk], 99.9)
        sun_candidates = (brightness_valid > threshold) & disk
        
        if sun_candidates.any():
            ys_sun, xs_sun = np.nonzero(sun_candidates)
            sun_x, sun_y = xs_sun.mean(), ys_sun.mean()
            sun_idx_y, sun_idx_x = int(sun_y), int(sun_x)
            sun_az_math = az_math[sun_idx_y, sun_idx_x]
            sun_ze_deg = ze_deg[sun_idx_y, sun_idx_x]
            sun_az_nav = (90.0 - sun_az_math) % 360.0
            print(f"  Detected: pixel=({sun_x:.1f}, {sun_y:.1f})")
            print(f"  Sun: az={sun_az_nav:.1f}° (nav), ze={sun_ze_deg:.1f}°")
        else:
            sun_az_nav, sun_ze_deg = MANUAL_SUN_AZ_NAV, MANUAL_SUN_ZE
            print(f"  WARNING: Detection failed, using manual values")
            print(f"  Sun: az={sun_az_nav:.1f}° (nav), ze={sun_ze_deg:.1f}°")
    else:
        sun_az_nav, sun_ze_deg = MANUAL_SUN_AZ_NAV, MANUAL_SUN_ZE
        print(f"\n[Using Manual Sun Position]")
        print(f"  Sun: az={sun_az_nav:.1f}° (nav), ze={sun_ze_deg:.1f}°")
    
    SPA_deg = compute_sun_pixel_angle(az_nav, ze_deg, sun_az_nav, sun_ze_deg)
    gamma = np.radians(SPA_deg)
    
    # Generate synthetic RGB channels (in linear space)
    print("\n[Generating Synthetic Clear-Sky]")
    rgb_syn_lin = np.zeros((H, W, 3), dtype=np.float32)
    
    for i, ch in enumerate(["R", "G", "B"]):
        A, B, C, D, E, F, H_coeff = coeffs[ch]
        G = G_model(theta, A, B, C, beta=BETA)
        S = S_model(gamma, D, E, F, H_coeff)
        L = G * S * EXPOSURE_SCALE
        rgb_syn_lin[..., i] = L
    
    rgb_syn_lin[~disk] = 0
    
    # Smooth color gradients with gentle Gaussian blur (reduces harsh transitions)
    # Apply blur before gamma encoding to work in linear space
    kernel_size = 5  # Adjust: larger = smoother (3, 5, 7, 9...)
    rgb_syn_lin_smooth = cv2.GaussianBlur(rgb_syn_lin, (kernel_size, kernel_size), sigmaX=1.5, sigmaY=1.5)
    # Restore mask (blur can leak into masked regions)
    rgb_syn_lin_smooth[~disk] = 0
    print(f"  Applied Gaussian blur (kernel={kernel_size}) for smooth color gradients")
    
    # Convert to sRGB if requested
    if GAMMA_ENCODE:
        rgb_syn = linear_to_srgb(rgb_syn_lin_smooth)
        print("  Converted to sRGB (gamma encoded)")
    else:
        rgb_syn = np.clip(rgb_syn_lin_smooth, 0, 1)
        print("  Kept as linear RGB")
    
    # Auto-scale to match real image brightness
    valid = disk & (ze_deg <= 85)
    if valid.sum() > 0:
        real_lin = srgb_to_linear(rgb_real)
        real_median = np.median(real_lin[valid])
        syn_median = np.median(rgb_syn_lin_smooth[valid])
        if syn_median > 1e-6:
            scale = real_median / syn_median
            rgb_syn_lin_smooth *= scale
            if GAMMA_ENCODE:
                rgb_syn = linear_to_srgb(rgb_syn_lin_smooth)
            else:
                rgb_syn = np.clip(rgb_syn_lin_smooth, 0, 1)
            print(f"  Auto-scaled by {scale:.3f}× to match real image")
    
    # Save synthetic image
    OUT_SYNTHETIC.parent.mkdir(parents=True, exist_ok=True)
    rgb_syn_uint8 = (rgb_syn * 255).astype(np.uint8)
    rgb_syn_masked = circular_mask(rgb_syn_uint8)
    Image.fromarray(rgb_syn_masked).save(OUT_SYNTHETIC)
    print(f"\n✅ Saved synthetic: {OUT_SYNTHETIC}")
    
    # Create comparison plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Real image
        rgb_real_uint8 = (rgb_real * 255).astype(np.uint8)
        rgb_real_masked = circular_mask(rgb_real_uint8).astype(np.float32) / 255.0
        axes[0].imshow(rgb_real_masked)
        axes[0].set_title("Real Clear-Sky Image", fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Synthetic image
        rgb_syn_display = rgb_syn_masked.astype(np.float32) / 255.0
        axes[1].imshow(rgb_syn_display)
        axes[1].set_title("Synthetic Clear-Sky (RGB Model)", fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Difference
        diff = rgb_real_masked.astype(np.float32)/255.0 - rgb_syn_display
        diff_mag = np.sqrt((diff**2).sum(axis=-1))
        im_diff = axes[2].imshow(diff_mag, cmap='hot', vmin=0, vmax=0.3)
        axes[2].set_title(f"Difference (RGB L2)\nMedian={np.median(diff_mag[valid]):.4f}", 
                         fontsize=14, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im_diff, ax=axes[2], fraction=0.046)
        
        plt.tight_layout()
        fig.savefig(OUT_COMPARISON, dpi=150)
        print(f"✅ Saved comparison: {OUT_COMPARISON}")
    except Exception as e:
        print(f"Comparison plot skipped: {e}")
    
    print("\n" + "="*70)
    print("STEP 2 COMPLETE!")
    print("\nOutputs:")
    print(f"  - Synthetic image: {OUT_SYNTHETIC}")
    print(f"  - Comparison plot: {OUT_COMPARISON}")
    print("\nTo generate clear-sky for a different sun position:")
    print("  1. Edit this script: set USE_DETECTED_SUN = False")
    print("  2. Set MANUAL_SUN_AZ_NAV and MANUAL_SUN_ZE to desired values")
    print("  3. Run again!")
    print("="*70)

if __name__ == "__main__":
    main()

