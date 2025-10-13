#!/usr/bin/env python3
"""
Generate synthetic clear-sky image for Mobotix camera using fitted coefficients.
"""

from pathlib import Path
import json
import numpy as np
from PIL import Image
import cv2
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from colorspace_utils import srgb_to_linear

# ----------------- CONFIG -----------------
BASE_DIR = Path(__file__).parent.parent
MOBOTIX_IMG = BASE_DIR / "input/mobotix_rgb.png"
SEMANTIC_MASK = BASE_DIR / "input/masks/mobotix_mask.png"
COEFFS_JSON = BASE_DIR / "output/mobotix/coefficients.json"

OUT_SYNTHETIC = BASE_DIR / "output/mobotix/synthetic_clearsky.png"
OUT_COMPARISON = BASE_DIR / "output/mobotix/comparison.png"

K_STEREO = 1.4  # K-tan stereographic projection
BETA = 0.32
EXPOSURE_SCALE = 1.0
GAMMA_ENCODE = True

# Sun size adjustment (optimized via grid search)
# Best match: E×0.70, F×1.20 → Combined diff: 0.0570
SUN_E_SCALE = 0.7   # Optimal E scale
SUN_F_SCALE = 1.2   # Optimal F scale

# Semantic mask classes
CLASS_FROM_RGB = {
    (66, 245, 84):  "background",
    (66, 135, 245): "clear_sky",
    (245, 66, 66):  "cloud",
    (245, 212, 66): "sun",
}

# ----------------- helpers -----------------
def k_tan_theta_from_radius(r, R, K=1.4):
    denom = np.tan(K * np.pi / 4.0)
    t = (r / R) * denom
    t = np.clip(t, 0.0, 1e6)
    return (2.0 / K) * np.arctan(t)

def equidistant_theta_from_radius(r, R):
    """Equidistant projection: r/R = θ/90° linearly."""
    return (r / R) * (np.pi / 2.0)

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

def G_model(theta_rad, A, B, C, beta=BETA):
    ct = np.clip(np.cos(theta_rad), 0.0, 1.0)
    return A * (1.0 + C * np.power(ct, beta)) / (1.0 + B * ct + 1e-8)

def S_model(gamma_rad, D, E, F, H):
    g = np.clip(gamma_rad, 1e-3, np.pi)
    return D + E * np.power(g, -F) + H * np.cos(gamma_rad)

def linear_to_srgb(linear_rgb):
    a = 0.055
    srgb = np.where(linear_rgb <= 0.0031308,
                    linear_rgb * 12.92,
                    (1 + a) * np.power(np.clip(linear_rgb, 0, None), 1/2.4) - a)
    return np.clip(srgb, 0, 1)

def fit_disk(mask_or_gray):
    """Estimate circle center and radius from semantic mask or grayscale."""
    from scipy.ndimage import binary_erosion
    
    a = mask_or_gray.astype(np.float32)
    if a.ndim == 3:
        a = a.mean(-1)
    
    # Threshold to get binary mask
    mask = a > (5.0 if a.max() > 1.5 else 0.02)
    H, W = a.shape
    
    if not mask.any():
        return (W-1)/2.0, (H-1)/2.0, min(W, H)/2.05
    
    # Center from mask centroid
    ys, xs = np.nonzero(mask)
    cx, cy = xs.mean(), ys.mean()
    
    # Radius from boundary pixels
    edge = mask & (~binary_erosion(mask, iterations=1))
    ys_e, xs_e = np.nonzero(edge)
    
    if xs_e.size < 30:
        R = min(W, H)/2.05
    else:
        R = np.median(np.hypot(xs_e - cx, ys_e - cy))
    
    return cx, cy, R

# ----------------- main -----------------
def main():
    print("="*70)
    print("Generating Mobotix synthetic clear-sky")
    print("="*70)
    
    # Load coefficients
    if not COEFFS_JSON.exists():
        print(f"ERROR: Coefficients not found. Run fit_mobotix_clearsky.py first!")
        return
    
    coeffs = json.loads(COEFFS_JSON.read_text())
    print(f"Loaded Mobotix coefficients from {COEFFS_JSON.name}")
    for ch in ["R", "G", "B"]:
        A,B,C,D,E,F,H = coeffs[ch]
        print(f"  {ch}: A={A:.4f} B={B:.4f} C={C:.4f}")
    
    # Load real image
    if not MOBOTIX_IMG.exists():
        print(f"ERROR: Image not found!")
        return
    
    img = Image.open(MOBOTIX_IMG).convert("RGB")
    rgb8_real = np.array(img, np.uint8)
    H, W = rgb8_real.shape[:2]
    print(f"\nImage: {W}×{H}")
    
    rgb_real = rgb8_real.astype(np.float32) / 255.0
    
    # Load semantic mask
    print("\n[Loading Semantic Mask]")
    class_masks = load_semantic_mask(SEMANTIC_MASK)
    sun_mask = class_masks.get("sun", np.zeros((H, W), dtype=bool))
    clear_sky_mask = class_masks.get("clear_sky", np.zeros((H, W), dtype=bool))
    print(f"  Clear-sky pixels: {clear_sky_mask.sum()} ({100*clear_sky_mask.sum()/(H*W):.1f}%)")
    
    # Detect sun
    print("\n[Detecting Sun Position]")
    if sun_mask.any():
        ys_sun, xs_sun = np.nonzero(sun_mask)
        sun_x = xs_sun.mean()
        sun_y = ys_sun.mean()
        print(f"  Sun pixel: ({sun_x:.1f}, {sun_y:.1f})")
    else:
        rgb_lin = srgb_to_linear(rgb_real)
        brightness = rgb_lin.mean(axis=-1)
        sun_idx = np.unravel_index(brightness.argmax(), brightness.shape)
        sun_y, sun_x = sun_idx
        print(f"  Sun pixel (fallback): ({sun_x:.1f}, {sun_y:.1f})")
    
    # Build geometry (fit disk from semantic mask + clear_sky + sun)
    print("\n[Building Geometry]")
    # Create combined sky mask (clear_sky + sun + clouds) to find circular boundary
    sky_combined = clear_sky_mask | sun_mask | class_masks.get("cloud", np.zeros((H, W), dtype=bool))
    cx, cy, R = fit_disk(sky_combined.astype(np.uint8) * 255)
    print(f"  Disk center: ({cx:.1f}, {cy:.1f}), radius: {R:.1f}px")
    
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
    
    # Sun angles
    sun_az_math = az_math[int(sun_y), int(sun_x)]
    sun_ze_deg = ze_deg[int(sun_y), int(sun_x)]
    sun_az_nav = (90.0 - sun_az_math) % 360.0
    print(f"  Sun: az={sun_az_nav:.1f}° (nav), ze={sun_ze_deg:.1f}°")
    
    SPA_deg = compute_sun_pixel_angle(az_nav, ze_deg, sun_az_nav, sun_ze_deg)
    gamma = np.radians(SPA_deg)
    
    # Create disk mask
    disk = r <= R
    
    # Generate synthetic RGB
    print("\n[Generating Synthetic Clear-Sky]")
    print(f"  Sun size adjustment: E×{SUN_E_SCALE:.2f}, F×{SUN_F_SCALE:.2f}")
    rgb_syn_lin = np.zeros((H, W, 3), dtype=np.float32)
    
    for i, ch in enumerate(["R", "G", "B"]):
        A, B, C, D, E, F, H_coeff = coeffs[ch]
        # Apply sun size scaling
        E_scaled = E * SUN_E_SCALE
        F_scaled = F * SUN_F_SCALE
        
        G = G_model(theta, A, B, C, beta=BETA)
        S = S_model(gamma, D, E_scaled, F_scaled, H_coeff)
        L = G * S * EXPOSURE_SCALE
        rgb_syn_lin[..., i] = L
    
    rgb_syn_lin[~disk] = 0
    
    # Smooth color gradients
    kernel_size = 7  # Larger kernel for 512x512
    rgb_syn_lin_smooth = cv2.GaussianBlur(rgb_syn_lin, (kernel_size, kernel_size), sigmaX=2.0, sigmaY=2.0)
    rgb_syn_lin_smooth[~disk] = 0
    print(f"  Applied Gaussian blur (kernel={kernel_size})")
    
    # Convert to sRGB
    if GAMMA_ENCODE:
        rgb_syn = linear_to_srgb(rgb_syn_lin_smooth)
        print("  Converted to sRGB (gamma encoded)")
    else:
        rgb_syn = np.clip(rgb_syn_lin_smooth, 0, 1)
        print("  Kept as linear RGB")
    
    # Auto-scale to match real image (using only clear-sky pixels from semantic mask)
    valid = clear_sky_mask & disk & (ze_deg <= 85)
    print(f"\n[Auto-scaling]")
    print(f"  Using {valid.sum()} clear-sky pixels for scaling")
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
            print(f"  Scaled by {scale:.3f}× to match real clear-sky brightness")
    
    # Save synthetic image
    OUT_SYNTHETIC.parent.mkdir(parents=True, exist_ok=True)
    rgb_syn_uint8 = (rgb_syn * 255).astype(np.uint8)
    Image.fromarray(rgb_syn_uint8).save(OUT_SYNTHETIC)
    print(f"\n✅ Saved synthetic: {OUT_SYNTHETIC}")
    
    # Create comparison plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Real image
        axes[0].imshow(rgb_real)
        axes[0].set_title("Real Mobotix Image", fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Synthetic image
        axes[1].imshow(rgb_syn)
        axes[1].set_title("Synthetic Clear-Sky (RGB Model)", fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Difference
        diff = rgb_real - rgb_syn
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
    print("MOBOTIX GENERATION COMPLETE!")
    print(f"\nOutputs:")
    print(f"  - Synthetic: {OUT_SYNTHETIC}")
    print(f"  - Comparison: {OUT_COMPARISON}")
    print("="*70)

if __name__ == "__main__":
    main()

