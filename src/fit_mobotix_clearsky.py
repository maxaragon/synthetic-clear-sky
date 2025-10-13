#!/usr/bin/env python3
"""
Fit clear-sky RGB coefficients for Mobotix camera using semantic mask.

Since we don't have calibration matrix, we:
1. Use sun detection as reference
2. Scale geometry from reference image (128x128 K-tan) to new size (512x512)
3. Fit only on "clear_sky" pixels from semantic mask
4. Save coefficients for Mobotix camera
"""

from pathlib import Path
import json
import numpy as np
from PIL import Image
import cv2
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from colorspace_utils import srgb_to_linear

from scipy.optimize import least_squares

# ----------------- CONFIG -----------------
MOBOTIX_IMG = Path("/Users/max/Desktop/Mobotix_20240613162000_512x512.jpg")
SEMANTIC_MASK = Path("/Users/max/Downloads/Mobotix_20240613162000_512x512_mask-2.png")

OUT_JSON = Path("/Users/max/Desktop/PhD/WUR/sky-transfer/synthetic-clearsky-rgb/output/parameters/clearsky_RGB_mobotix.json")
OUT_PLOT = Path("/Users/max/Desktop/PhD/WUR/sky-transfer/synthetic-clearsky-rgb/output/diagnostics/fit_diagnostics_mobotix.png")

# Semantic mask classes (RGB values)
CLASS_FROM_RGB = {
    (66, 245, 84):  "background",
    (66, 135, 245): "clear_sky",
    (245, 66, 66):  "cloud",
    (245, 212, 66): "sun",
}

# Geometry: K-tan stereographic projection
K_STEREO = 1.4
BETA = 0.32

# Exclusions
BAND_GAMMA_MIN = 88.0
BAND_GAMMA_MAX = 92.0
SAT_THRESH = 0.98

# ----------------- helpers -----------------
def k_tan_theta_from_radius(r, R, K=1.4):
    """Inverse K-tan stereographic: radius r → zenith θ (radians)."""
    denom = np.tan(K * np.pi / 4.0)
    t = (r / R) * denom
    t = np.clip(t, 0.0, 1e6)
    return (2.0 / K) * np.arctan(t)

def equidistant_theta_from_radius(r, R):
    """Equidistant projection: r/R = θ/90° linearly."""
    return (r / R) * (np.pi / 2.0)

def compute_sun_pixel_angle(az_pix, ze_pix, az_sun, ze_sun):
    """Compute sun-pixel angle (SPA) in degrees."""
    # Convert to radians
    az_pix_rad = np.radians(az_pix)
    ze_pix_rad = np.radians(ze_pix)
    az_sun_rad = np.radians(az_sun)
    ze_sun_rad = np.radians(ze_sun)
    
    # Spherical law of cosines
    cos_spa = (np.sin(ze_pix_rad) * np.sin(ze_sun_rad) * np.cos(az_pix_rad - az_sun_rad) +
               np.cos(ze_pix_rad) * np.cos(ze_sun_rad))
    cos_spa = np.clip(cos_spa, -1.0, 1.0)
    spa_rad = np.arccos(cos_spa)
    return np.degrees(spa_rad)

def load_semantic_mask(mask_path):
    """Load semantic mask and return dict of class masks."""
    mask_img = Image.open(mask_path).convert("RGB")
    mask_arr = np.array(mask_img, np.uint8)
    H, W = mask_arr.shape[:2]
    
    class_masks = {}
    for rgb, class_name in CLASS_FROM_RGB.items():
        # Match RGB values
        match = np.all(mask_arr == rgb, axis=-1)
        class_masks[class_name] = match
        print(f"  {class_name:15s}: {match.sum():6d} pixels ({100*match.sum()/(H*W):5.2f}%)")
    
    return class_masks

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
    print("Fitting Mobotix clear-sky RGB coefficients")
    print("="*70)
    
    # 1) Load image
    if not MOBOTIX_IMG.exists():
        print(f"ERROR: Image not found: {MOBOTIX_IMG}")
        return
    
    img = Image.open(MOBOTIX_IMG).convert("RGB")
    rgb8 = np.array(img, np.uint8)
    H, W = rgb8.shape[:2]
    print(f"Image: {W}×{H}")
    
    rgb = rgb8.astype(np.float32) / 255.0
    
    # 2) Load semantic mask
    if not SEMANTIC_MASK.exists():
        print(f"ERROR: Semantic mask not found: {SEMANTIC_MASK}")
        return
    
    print("\n[Loading Semantic Mask]")
    class_masks = load_semantic_mask(SEMANTIC_MASK)
    
    clear_sky_mask = class_masks.get("clear_sky", np.zeros((H, W), dtype=bool))
    sun_mask = class_masks.get("sun", np.zeros((H, W), dtype=bool))
    
    if not clear_sky_mask.any():
        print("ERROR: No clear_sky pixels found in mask!")
        return
    
    # 3) Detect sun position from sun mask
    print("\n[Detecting Sun Position]")
    if sun_mask.any():
        ys_sun, xs_sun = np.nonzero(sun_mask)
        sun_x = xs_sun.mean()
        sun_y = ys_sun.mean()
        print(f"  Sun pixel (from mask): ({sun_x:.1f}, {sun_y:.1f})")
    else:
        # Fallback: brightest pixel
        rgb_lin = srgb_to_linear(rgb)
        brightness = rgb_lin.mean(axis=-1)
        sun_idx = np.unravel_index(brightness.argmax(), brightness.shape)
        sun_y, sun_x = sun_idx
        print(f"  Sun pixel (from brightness): ({sun_x:.1f}, {sun_y:.1f})")
    
    # 4) Build geometry: fit disk from semantic mask
    print("\n[Building Geometry]")
    # Create combined sky mask (clear_sky + sun + clouds) to find circular boundary
    sky_combined = clear_sky_mask | sun_mask | class_masks.get("cloud", np.zeros((H, W), dtype=bool))
    cx, cy, R = fit_disk(sky_combined.astype(np.uint8) * 255)
    print(f"  Fitted disk center: ({cx:.1f}, {cy:.1f})")
    print(f"  Fitted disk radius: {R:.1f}px")
    
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    x = xx - cx
    y = cy - yy  # y up
    r = np.hypot(x, y)
    r_clipped = np.minimum(r, R)
    
    # Convert to zenith angle using K-tan stereographic
    theta = k_tan_theta_from_radius(r_clipped, R, K=K_STEREO)
    ze_deg = np.degrees(theta)
    
    # Azimuth (mathematical convention: 0° at +x, CCW)
    phi = np.arctan2(y, x)
    az_math = (np.degrees(phi) % 360.0).astype(np.float32)
    az_nav = (90.0 - az_math) % 360.0  # Navigation: 0° at North, CW
    
    # Sun angles from sun pixel
    sun_az_math = az_math[int(sun_y), int(sun_x)]
    sun_ze_deg = ze_deg[int(sun_y), int(sun_x)]
    sun_az_nav = (90.0 - sun_az_math) % 360.0
    print(f"  Sun position: az={sun_az_nav:.1f}° (nav), ze={sun_ze_deg:.1f}°")
    
    # Compute SPA
    SPA_deg = compute_sun_pixel_angle(az_nav, ze_deg, sun_az_nav, sun_ze_deg)
    gamma = np.radians(SPA_deg)
    
    # 5) Define fitting mask: clear_sky pixels only
    print("\n[Building Fitting Mask]")
    fit_mask = clear_sky_mask.copy()
    print(f"  Base clear_sky pixels: {fit_mask.sum()}")
    
    # Optional: exclude near-horizon for stability
    fit_mask &= (ze_deg <= 85.0)
    print(f"  After horizon cutoff: {fit_mask.sum()}")
    
    # 6) Convert to linear RGB
    rgb_lin = srgb_to_linear(rgb)
    print("\n[Converted to linear RGB]")
    
    # 7) Fit per channel
    coeffs = {}
    chan_names = ["R", "G", "B"]
    chans = [rgb_lin[...,0], rgb_lin[...,1], rgb_lin[...,2]]
    
    # Band for gradation fit (γ ≈ 90°)
    band = fit_mask & (SPA_deg >= BAND_GAMMA_MIN) & (SPA_deg <= BAND_GAMMA_MAX)
    print(f"\n[Gradation band (γ≈90°)]: {band.sum()} pixels")
    
    print("\n[Fitting Chauvin Model per Channel]")
    for ch_name, I in zip(chan_names, chans):
        I = I.copy()
        
        # Drop saturated tail
        sat_cut = percent_clip(I[fit_mask], SAT_THRESH*100.0)
        valid = fit_mask & (I <= sat_cut)
        print(f"\n  {ch_name}: {valid.sum()} valid pixels")
        
        # Fit G(θ)
        th_band = theta[band]
        I_band = I[band]
        if I_band.size < 100:
            print(f"    WARNING: Insufficient band pixels, using all valid")
            th_band = theta[valid]
            I_band = I[valid]
        
        A0, B0, C0 = max(np.median(I_band), 1e-3), -0.5, 0.2
        bounds_G = ([0.0, -2.0, -2.0], [np.inf, 0.0, 1.5])
        def resid_G(params):
            A, B, C = params
            return (G_model(th_band, A,B,C) - I_band)
        sol_G = least_squares(resid_G, x0=[A0,B0,C0], bounds=bounds_G, loss='huber', f_scale=0.02)
        A, B, C = sol_G.x.tolist()
        
        # Fit S(γ)
        G_full = G_model(theta, A,B,C)
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
            D,E,F,H = params
            return (S_model(g_fit, D,E,F,H) - s_fit)
        sol_S = least_squares(resid_S, x0=[D0,E0,F0,H0], bounds=bounds_S, loss='huber', f_scale=0.02)
        D,E,F,H = sol_S.x.tolist()
        
        coeffs[ch_name] = [float(A), float(B), float(C), float(D), float(E), float(F), float(H)]
        print(f"    A={A:.4f} B={B:.4f} C={C:.4f} D={D:.4f} E={E:.4f} F={F:.4f} H={H:.4f}")
    
    # 8) Save JSON
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(coeffs, indent=2))
    print(f"\n✅ Saved Mobotix coefficients: {OUT_JSON}")
    
    # 9) Diagnostic plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 3, figsize=(14, 12))
        theta_deg = np.degrees(theta)
        
        for i,(ch_name,I) in enumerate(zip(chan_names, chans)):
            A,B,C,D,E,F,H = coeffs[ch_name]
            
            # Channel image
            im_disp = (np.clip(I / percent_clip(I[fit_mask],98),0,1)*255).astype(np.uint8)
            axes[i,0].imshow(im_disp, cmap='gray')
            axes[i,0].set_title(f"{ch_name} channel"); axes[i,0].axis('off')
            
            # G vs theta
            band_mask = (SPA_deg >= BAND_GAMMA_MIN) & (SPA_deg <= BAND_GAMMA_MAX) & fit_mask
            tband = theta_deg[band_mask]; Iband = I[band_mask]
            axes[i,1].scatter(tband, Iband, s=4, alpha=0.2)
            tgrid = np.linspace(0, 90, 181)
            axes[i,1].plot(tgrid, G_model(np.radians(tgrid), A,B,C), 'r-', lw=2)
            axes[i,1].set_xlabel("θ (deg)"); axes[i,1].set_ylabel(f"{ch_name}")
            axes[i,1].set_title(f"Gradation G(θ)"); axes[i,1].grid(True, alpha=0.3)
            
            # S vs gamma
            valid = fit_mask & (I <= percent_clip(I[fit_mask],98))
            g_fit = np.degrees(gamma[valid]); s_fit = np.clip((I/G_model(theta,A,B,C))[valid], 1e-6, None)
            axes[i,2].scatter(g_fit, s_fit, s=4, alpha=0.2)
            ggrid = np.linspace(5, 180, 351)
            axes[i,2].plot(ggrid, S_model(np.radians(ggrid), D,E,F,H), 'r-', lw=2)
            axes[i,2].set_xlabel("γ (deg)"); axes[i,2].set_ylabel("S")
            axes[i,2].set_title(f"Scattering S(γ)"); axes[i,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_PLOT, dpi=160)
        print(f"✅ Saved diagnostics: {OUT_PLOT}")
    except Exception as e:
        print(f"Diagnostic plot skipped: {e}")
    
    print("\n" + "="*70)
    print("MOBOTIX FITTING COMPLETE!")
    print(f"Coefficients: {OUT_JSON}")
    print(f"Diagnostics:  {OUT_PLOT}")
    print("="*70)

if __name__ == "__main__":
    main()

